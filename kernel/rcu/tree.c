/*
 * Read-Copy Update mechanism for mutual exclusion (tree-based version)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, you can access it online at
 * http://www.gnu.org/licenses/gpl-2.0.html.
 *
 * Copyright IBM Corporation, 2008
 *
 * Authors: Dipankar Sarma <dipankar@in.ibm.com>
 *	    Manfred Spraul <manfred@colorfullife.com>
 *	    Paul E. McKenney <paulmck@linux.ibm.com>
 *
 * Based on the original work by Paul McKenney <paulmck@us.ibm.com>
 * and inputs from Rusty Russell, Andrea Arcangeli and Andi Kleen.
 *
 * For detailed explanation of Read-Copy Update mechanism see -
 *	Documentation/RCU
 */

#define pr_fmt(fmt) "rcu: " fmt

#include <linux/types.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/spinlock.h>
#include <linux/smp.h>
#include <linux/rcupdate_wait.h>
#include <linux/interrupt.h>
#include <linux/sched.h>
#include <linux/sched/debug.h>
#include <linux/nmi.h>
#include <linux/atomic.h>
#include <linux/bitops.h>
#include <linux/export.h>
#include <linux/completion.h>
#include <linux/moduleparam.h>
#include <linux/percpu.h>
#include <linux/notifier.h>
#include <linux/cpu.h>
#include <linux/mutex.h>
#include <linux/time.h>
#include <linux/kernel_stat.h>
#include <linux/wait.h>
#include <linux/kthread.h>
#include <uapi/linux/sched/types.h>
#include <linux/prefetch.h>
#include <linux/delay.h>
#include <linux/random.h>
#include <linux/trace_events.h>
#include <linux/suspend.h>
#include <linux/ftrace.h>
#include <linux/tick.h>
#include <linux/sysrq.h>
#include <linux/kprobes.h>
#include <linux/gfp.h>
#include <linux/oom.h>
#include <linux/smpboot.h>
#include <linux/jiffies.h>
#include <linux/slab.h>
#include <linux/sched/isolation.h>
#include <linux/sched/clock.h>
#include "../time/tick-internal.h"

#include "tree.h"
#include "rcu.h"

#ifdef MODULE_PARAM_PREFIX
#undef MODULE_PARAM_PREFIX
#endif
#define MODULE_PARAM_PREFIX "rcutree."

#ifndef data_race
#define data_race(expr)							\
	({								\
		expr;							\
	})
#endif
#ifndef ASSERT_EXCLUSIVE_WRITER
#define ASSERT_EXCLUSIVE_WRITER(var) do { } while (0)
#endif
#ifndef ASSERT_EXCLUSIVE_ACCESS
#define ASSERT_EXCLUSIVE_ACCESS(var) do { } while (0)
#endif

/* Data structures. */

/*
 * In order to export the rcu_state name to the tracing tools, it
 * needs to be added in the __tracepoint_string section.
 * This requires defining a separate variable tp_<sname>_varname
 * that points to the string being used, and this will allow
 * the tracing userspace tools to be able to decipher the string
 * address to the matching string.
 */
#define RCU_DYNTICK_CTRL_MASK 0x1
#define RCU_DYNTICK_CTRL_CTR  (RCU_DYNTICK_CTRL_MASK + 1)

static DEFINE_PER_CPU_SHARED_ALIGNED(struct rcu_data, rcu_data) = {
	.dynticks_nesting = 1,
	.dynticks_nmi_nesting = DYNTICK_IRQ_NONIDLE,
	.dynticks = ATOMIC_INIT(RCU_DYNTICK_CTRL_CTR),
};
static struct rcu_state rcu_state = {
	.level = { &rcu_state.node[0] },
	.gp_state = RCU_GP_IDLE,
	.gp_seq = (0UL - 300UL) << RCU_SEQ_CTR_SHIFT,
	.barrier_mutex = __MUTEX_INITIALIZER(rcu_state.barrier_mutex),
	.name = RCU_NAME,
	.abbr = RCU_ABBR,
	.exp_mutex = __MUTEX_INITIALIZER(rcu_state.exp_mutex),
	.exp_wake_mutex = __MUTEX_INITIALIZER(rcu_state.exp_wake_mutex),
	.ofl_lock = __RAW_SPIN_LOCK_UNLOCKED(rcu_state.ofl_lock),
};

/* Dump rcu_node combining tree at boot to verify correct setup. */
static bool dump_tree;
module_param(dump_tree, bool, 0444);
/* By default, use RCU_SOFTIRQ instead of rcuc kthreads. */
static bool use_softirq = true;
module_param(use_softirq, bool, 0444);
/* Control rcu_node-tree auto-balancing at boot time. */
static bool rcu_fanout_exact;
module_param(rcu_fanout_exact, bool, 0444);
/* Increase (but not decrease) the RCU_FANOUT_LEAF at boot time. */
static int rcu_fanout_leaf = RCU_FANOUT_LEAF;
module_param(rcu_fanout_leaf, int, 0444);
int rcu_num_lvls __read_mostly = RCU_NUM_LVLS;
/* Number of rcu_nodes at specified level. */
int num_rcu_lvl[] = NUM_RCU_LVL_INIT;
int rcu_num_nodes __read_mostly = NUM_RCU_NODES; /* Total # rcu_nodes in use. */
/* panic() on RCU Stall sysctl. */
int sysctl_panic_on_rcu_stall __read_mostly = CONFIG_RCU_PANIC_ON_STALL;

/*
 * The rcu_scheduler_active variable is initialized to the value
 * RCU_SCHEDULER_INACTIVE and transitions RCU_SCHEDULER_INIT just before the
 * first task is spawned.  So when this variable is RCU_SCHEDULER_INACTIVE,
 * RCU can assume that there is but one task, allowing RCU to (for example)
 * optimize synchronize_rcu() to a simple barrier().  When this variable
 * is RCU_SCHEDULER_INIT, RCU must actually do all the hard work required
 * to detect real grace periods.  This variable is also used to suppress
 * boot-time false positives from lockdep-RCU error checking.  Finally, it
 * transitions from RCU_SCHEDULER_INIT to RCU_SCHEDULER_RUNNING after RCU
 * is fully initialized, including all of its kthreads having been spawned.
 */
int rcu_scheduler_active __read_mostly;
EXPORT_SYMBOL_GPL(rcu_scheduler_active);

/*
 * The rcu_scheduler_fully_active variable transitions from zero to one
 * during the early_initcall() processing, which is after the scheduler
 * is capable of creating new tasks.  So RCU processing (for example,
 * creating tasks for RCU priority boosting) must be delayed until after
 * rcu_scheduler_fully_active transitions from zero to one.  We also
 * currently delay invocation of any RCU callbacks until after this point.
 *
 * It might later prove better for people registering RCU callbacks during
 * early boot to take responsibility for these callbacks, but one step at
 * a time.
 */
static int rcu_scheduler_fully_active __read_mostly;

static void
rcu_report_qs_rnp(unsigned long mask, struct rcu_state *rsp,
		  struct rcu_node *rnp, unsigned long gps, unsigned long flags);
static void rcu_init_new_rnp(struct rcu_node *rnp_leaf);
static void rcu_cleanup_dead_rnp(struct rcu_node *rnp_leaf);
static void rcu_boost_kthread_setaffinity(struct rcu_node *rnp, int outgoingcpu);
static void invoke_rcu_core(void);
static void invoke_rcu_callbacks(struct rcu_state *rsp, struct rcu_data *rdp);
static void rcu_report_exp_rdp(struct rcu_state *rsp,
			       struct rcu_data *rdp, bool wake);
static void sync_sched_exp_online_cleanup(int cpu);
static void check_cb_ovld_locked(struct rcu_data *rdp, struct rcu_node *rnp);

/* rcuc/rcub kthread realtime priority */
static int kthread_prio = IS_ENABLED(CONFIG_RCU_BOOST) ? 1 : 0;
module_param(kthread_prio, int, 0644);

/* Delay in jiffies for grace-period initialization delays, debug only. */

static int gp_preinit_delay;
module_param(gp_preinit_delay, int, 0444);
static int gp_init_delay;
module_param(gp_init_delay, int, 0444);
static int gp_cleanup_delay;
module_param(gp_cleanup_delay, int, 0444);

/*
 * This rcu parameter is runtime-read-only. It reflects
 * a minimum allowed number of objects which can be cached
 * per-CPU. Object size is equal to one page. This value
 * can be changed at boot time.
 */
static int rcu_min_cached_objs = 2;
module_param(rcu_min_cached_objs, int, 0444);

/* Retrieve RCU kthreads priority for rcutorture */
int rcu_get_gp_kthreads_prio(void)
{
	return kthread_prio;
}
EXPORT_SYMBOL_GPL(rcu_get_gp_kthreads_prio);

/*
 * Number of grace periods between delays, normalized by the duration of
 * the delay.  The longer the delay, the more the grace periods between
 * each delay.  The reason for this normalization is that it means that,
 * for non-zero delays, the overall slowdown of grace periods is constant
 * regardless of the duration of the delay.  This arrangement balances
 * the need for long delays to increase some race probabilities with the
 * need for fast grace periods to increase other race probabilities.
 */
#define PER_RCU_NODE_PERIOD 3	/* Number of grace periods between delays. */

/*
 * Compute the mask of online CPUs for the specified rcu_node structure.
 * This will not be stable unless the rcu_node structure's ->lock is
 * held, but the bit corresponding to the current CPU will be stable
 * in most contexts.
 */
static unsigned long rcu_rnp_online_cpus(struct rcu_node *rnp)
{
	return READ_ONCE(rnp->qsmaskinitnext);
}

/*
 * Return true if an RCU grace period is in progress.  The READ_ONCE()s
 * permit this function to be invoked without holding the root rcu_node
 * structure's ->lock, but of course results can be subject to change.
 */
static int rcu_gp_in_progress(struct rcu_state *rsp)
{
	return rcu_seq_state(rcu_seq_current(&rsp->gp_seq));
}

/*
 * Note a quiescent state.  Because we do not need to know
 * how many quiescent states passed, just if there was at least
 * one since the start of the grace period, this just sets a flag.
 * The caller must have disabled preemption.
 */
void rcu_sched_qs(void)
{
	RCU_LOCKDEP_WARN(preemptible(), "rcu_sched_qs() invoked with preemption enabled!!!");
	if (!__this_cpu_read(rcu_sched_data.cpu_no_qs.s))
		return;
	trace_rcu_grace_period(TPS("rcu_sched"),
			       __this_cpu_read(rcu_sched_data.gp_seq),
			       TPS("cpuqs"));
	__this_cpu_write(rcu_sched_data.cpu_no_qs.b.norm, false);
	if (!__this_cpu_read(rcu_sched_data.cpu_no_qs.b.exp))
		return;
	__this_cpu_write(rcu_sched_data.cpu_no_qs.b.exp, false);
	rcu_report_exp_rdp(&rcu_sched_state,
			   this_cpu_ptr(&rcu_sched_data), true);
}

void rcu_bh_qs(void)
{
	RCU_LOCKDEP_WARN(preemptible(), "rcu_bh_qs() invoked with preemption enabled!!!");
	if (__this_cpu_read(rcu_bh_data.cpu_no_qs.s)) {
		trace_rcu_grace_period(TPS("rcu_bh"),
				       __this_cpu_read(rcu_bh_data.gp_seq),
				       TPS("cpuqs"));
		__this_cpu_write(rcu_bh_data.cpu_no_qs.b.norm, false);
	}
}

/*
 * Steal a bit from the bottom of ->dynticks for idle entry/exit
 * control.  Initially this is for TLB flushing.
 */
#define RCU_DYNTICK_CTRL_MASK 0x1
#define RCU_DYNTICK_CTRL_CTR  (RCU_DYNTICK_CTRL_MASK + 1)
#ifndef rcu_eqs_special_exit
#define rcu_eqs_special_exit() do { } while (0)
#endif

static DEFINE_PER_CPU(struct rcu_dynticks, rcu_dynticks) = {
	.dynticks_nesting = 1,
	.dynticks_nmi_nesting = DYNTICK_IRQ_NONIDLE,
	.dynticks = ATOMIC_INIT(RCU_DYNTICK_CTRL_CTR),
};

/*
 * Record entry into an extended quiescent state.  This is only to be
 * called when not already in an extended quiescent state, that is,
 * RCU is watching prior to the call to this function and is no longer
 * watching upon return.
 */
static noinstr void rcu_dynticks_eqs_enter(void)
{
	struct rcu_dynticks *rdtp = this_cpu_ptr(&rcu_dynticks);
	int seq;

	/*
	 * CPUs seeing atomic_add_return() must see prior RCU read-side
	 * critical sections, and we also must force ordering with the
	 * next idle sojourn.
	 */
	rcu_dynticks_task_trace_enter();  // Before ->dynticks update!
	seq = arch_atomic_add_return(RCU_DYNTICK_CTRL_CTR, &rdp->dynticks);
	// RCU is no longer watching.  Better be in extended quiescent state!
	WARN_ON_ONCE(IS_ENABLED(CONFIG_RCU_EQS_DEBUG) &&
		     (seq & RCU_DYNTICK_CTRL_CTR));
	/* Better not have special action (TLB flush) pending! */
	WARN_ON_ONCE(IS_ENABLED(CONFIG_RCU_EQS_DEBUG) &&
		     (seq & RCU_DYNTICK_CTRL_MASK));
}

/*
 * Record exit from an extended quiescent state.  This is only to be
 * called from an extended quiescent state, that is, RCU is not watching
 * prior to the call to this function and is watching upon return.
 */
static noinstr void rcu_dynticks_eqs_exit(void)
{
	struct rcu_dynticks *rdtp = this_cpu_ptr(&rcu_dynticks);
	int seq;

	/*
	 * CPUs seeing atomic_add_return() must see prior idle sojourns,
	 * and we also must force ordering with the next RCU read-side
	 * critical section.
	 */
	seq = arch_atomic_add_return(RCU_DYNTICK_CTRL_CTR, &rdp->dynticks);
	// RCU is now watching.  Better not be in an extended quiescent state!
	rcu_dynticks_task_trace_exit();  // After ->dynticks update!
	WARN_ON_ONCE(IS_ENABLED(CONFIG_RCU_EQS_DEBUG) &&
		     !(seq & RCU_DYNTICK_CTRL_CTR));
	if (seq & RCU_DYNTICK_CTRL_MASK) {
		arch_atomic_andnot(RCU_DYNTICK_CTRL_MASK, &rdp->dynticks);
		smp_mb__after_atomic(); /* _exit after clearing mask. */
	}
}

/*
 * Reset the current CPU's ->dynticks counter to indicate that the
 * newly onlined CPU is no longer in an extended quiescent state.
 * This will either leave the counter unchanged, or increment it
 * to the next non-quiescent value.
 *
 * The non-atomic test/increment sequence works because the upper bits
 * of the ->dynticks counter are manipulated only by the corresponding CPU,
 * or when the corresponding CPU is offline.
 */
static void rcu_dynticks_eqs_online(void)
{
	struct rcu_dynticks *rdtp = this_cpu_ptr(&rcu_dynticks);

	if (atomic_read(&rdtp->dynticks) & RCU_DYNTICK_CTRL_CTR)
		return;
	atomic_add(RCU_DYNTICK_CTRL_CTR, &rdtp->dynticks);
}

/*
 * Is the current CPU in an extended quiescent state?
 *
 * No ordering, as we are sampling CPU-local information.
 */
static __always_inline bool rcu_dynticks_curr_cpu_in_eqs(void)
{
	struct rcu_dynticks *rdtp = this_cpu_ptr(&rcu_dynticks);

	return !(arch_atomic_read(&rdp->dynticks) & RCU_DYNTICK_CTRL_CTR);
}

/*
 * Snapshot the ->dynticks counter with full ordering so as to allow
 * stable comparison of this counter with past and future snapshots.
 */
static int rcu_dynticks_snap(struct rcu_data *rdp)
{
	int snap = atomic_add_return(0, &rdtp->dynticks);

	return snap & ~RCU_DYNTICK_CTRL_MASK;
}

/*
 * Return true if the snapshot returned from rcu_dynticks_snap()
 * indicates that RCU is in an extended quiescent state.
 */
static bool rcu_dynticks_in_eqs(int snap)
{
	return !(snap & RCU_DYNTICK_CTRL_CTR);
}

/*
 * Return true if the CPU corresponding to the specified rcu_dynticks
 * structure has spent some time in an extended quiescent state since
 * rcu_dynticks_snap() returned the specified snapshot.
 */
static bool rcu_dynticks_in_eqs_since(struct rcu_dynticks *rdtp, int snap)
{
	return snap != rcu_dynticks_snap(rdtp);
}

/*
 * Return true if the referenced integer is zero while the specified
 * CPU remains within a single extended quiescent state.
 */
bool rcu_dynticks_zero_in_eqs(int cpu, int *vp)
{
	struct rcu_data *rdp = per_cpu_ptr(&rcu_data, cpu);
	int snap;

	// If not quiescent, force back to earlier extended quiescent state.
	snap = atomic_read(&rdp->dynticks) & ~(RCU_DYNTICK_CTRL_MASK |
					       RCU_DYNTICK_CTRL_CTR);

	smp_rmb(); // Order ->dynticks and *vp reads.
	if (READ_ONCE(*vp))
		return false;  // Non-zero, so report failure;
	smp_rmb(); // Order *vp read and ->dynticks re-read.

	// If still in the same extended quiescent state, we are good!
	return snap == (atomic_read(&rdp->dynticks) & ~RCU_DYNTICK_CTRL_MASK);
}

/*
 * Set the special (bottom) bit of the specified CPU so that it
 * will take special action (such as flushing its TLB) on the
 * next exit from an extended quiescent state.  Returns true if
 * the bit was successfully set, or false if the CPU was not in
 * an extended quiescent state.
 */
bool rcu_eqs_special_set(int cpu)
{
	int old;
	int new;
	int new_old;
	struct rcu_data *rdp = &per_cpu(rcu_data, cpu);

	new_old = atomic_read(&rdp->dynticks);
	do {
<<<<<<< HEAD
		old = atomic_read(&rdtp->dynticks);
		if (old & RCU_DYNTICK_CTRL_CTR)
			return false;
		new_old = atomic_cmpxchg(&rdp->dynticks, old, new);
	} while (new_old != old);
>>>>>>> df84e4d09c29 (rcu: Optimize and protect atomic_cmpxchg() loop)
	return true;
}

 * Let the RCU core know that this CPU has gone through the scheduler,
 * which is a quiescent state.  This is called when the need for a
 * quiescent state is urgent, so we burn an atomic operation and full
 * memory barriers to let the RCU core know about it, regardless of what
 * this CPU might (or might not) do in the near future.
 *
 * We inform the RCU core by emulating a zero-duration dyntick-idle period.
 *
 * The caller must have disabled interrupts and must not be idle.
 */
void rcu_momentary_dyntick_idle(void)
{
	struct rcu_dynticks *rdtp = this_cpu_ptr(&rcu_dynticks);
	int special;

	raw_cpu_write(rcu_dynticks.rcu_need_heavy_qs, false);
	special = atomic_add_return(2 * RCU_DYNTICK_CTRL_CTR, &rdtp->dynticks);
	/* It is illegal to call this from idle state. */
	WARN_ON_ONCE(!(special & RCU_DYNTICK_CTRL_CTR));
}
EXPORT_SYMBOL_GPL(rcu_momentary_dyntick_idle);

/**
 * rcu_is_cpu_rrupt_from_idle - see if 'interrupted' from idle
 *
 * If the current CPU is idle and running at a first-level (not nested)
 * interrupt, or directly, from idle, return true.
 *
 * The caller must have at least disabled IRQs.
 */
void rcu_note_context_switch(bool preempt)
{
	long nesting;

	/*
	 * Usually called from the tick; but also used from smp_function_call()
	 * for expedited grace periods. This latter can result in running from
	 * the idle task, instead of an actual IPI.
	 */
	lockdep_assert_irqs_disabled();

	/* Check for counter underflows */
	RCU_LOCKDEP_WARN(__this_cpu_read(rcu_data.dynticks_nesting) < 0,
			 "RCU dynticks_nesting counter underflow!");
	RCU_LOCKDEP_WARN(__this_cpu_read(rcu_data.dynticks_nmi_nesting) <= 0,
			 "RCU dynticks_nmi_nesting counter underflow/zero!");

	/* Are we at first interrupt nesting level? */
	nesting = __this_cpu_read(rcu_data.dynticks_nmi_nesting);
	if (nesting > 1)
		return false;

	/*
	 * If we're not in an interrupt, we must be in the idle task!
	 */
	WARN_ON_ONCE(!nesting && !is_idle_task(current));

	/* Does CPU appear to be idle from an RCU standpoint? */
	return __this_cpu_read(rcu_data.dynticks_nesting) == 0;
}
EXPORT_SYMBOL_GPL(rcu_note_context_switch);

/*
 * Register a quiescent state for all RCU flavors.  If there is an
 * emergency, invoke rcu_momentary_dyntick_idle() to do a heavy-weight
 * dyntick-idle quiescent state visible to other CPUs (but only for those
 * RCU flavors in desperate need of a quiescent state, which will normally
 * be none of them).  Either way, do a lightweight quiescent state for
 * all RCU flavors.
 *
 * The barrier() calls are redundant in the common case when this is
 * called externally, but just in case this is called from within this
 * file.
 *
 */
void rcu_all_qs(void)
{
	unsigned long flags;

	if (!raw_cpu_read(rcu_dynticks.rcu_urgent_qs))
		return;
	preempt_disable();
	/* Load rcu_urgent_qs before other flags. */
	if (!smp_load_acquire(this_cpu_ptr(&rcu_dynticks.rcu_urgent_qs))) {
		preempt_enable();
		return;
	}
	this_cpu_write(rcu_dynticks.rcu_urgent_qs, false);
	barrier(); /* Avoid RCU read-side critical sections leaking down. */
	if (unlikely(raw_cpu_read(rcu_dynticks.rcu_need_heavy_qs))) {
		local_irq_save(flags);
		rcu_momentary_dyntick_idle();
		local_irq_restore(flags);
	}
	if (unlikely(raw_cpu_read(rcu_sched_data.cpu_no_qs.b.exp)))
		rcu_sched_qs();
	this_cpu_inc(rcu_dynticks.rcu_qs_ctr);
	barrier(); /* Avoid RCU read-side critical sections leaking up. */
	preempt_enable();
}
EXPORT_SYMBOL_GPL(rcu_all_qs);

#define DEFAULT_RCU_BLIMIT 10     /* Maximum callbacks per rcu_do_batch. */
static long blimit = DEFAULT_RCU_BLIMIT;
#define DEFAULT_RCU_QHIMARK 10000 /* If this many pending, ignore blimit. */
static long qhimark = DEFAULT_RCU_QHIMARK;
#define DEFAULT_RCU_QLOMARK 100   /* Once only this many pending, use blimit. */
static long qlowmark = DEFAULT_RCU_QLOMARK;
#define DEFAULT_RCU_QOVLD_MULT 2
#define DEFAULT_RCU_QOVLD (DEFAULT_RCU_QOVLD_MULT * DEFAULT_RCU_QHIMARK)
static long qovld = DEFAULT_RCU_QOVLD; /* If this many pending, hammer QS. */
static long qovld_calc = -1;	  /* No pre-initialization lock acquisitions! */

module_param(blimit, long, 0444);
module_param(qhimark, long, 0444);
module_param(qlowmark, long, 0444);
module_param(qovld, long, 0444);

static ulong jiffies_till_first_fqs = ULONG_MAX;
static ulong jiffies_till_next_fqs = ULONG_MAX;
static bool rcu_kick_kthreads;

static int param_set_first_fqs_jiffies(const char *val, const struct kernel_param *kp)
{
	ulong j;
	int ret = kstrtoul(val, 0, &j);

	if (!ret)
		WRITE_ONCE(*(ulong *)kp->arg, (j > HZ) ? HZ : j);
	return ret;
}

static int param_set_next_fqs_jiffies(const char *val, const struct kernel_param *kp)
{
	ulong j;
	int ret = kstrtoul(val, 0, &j);

	if (!ret)
		WRITE_ONCE(*(ulong *)kp->arg, (j > HZ) ? HZ : (j ?: 1));
	return ret;
}

static struct kernel_param_ops first_fqs_jiffies_ops = {
	.set = param_set_first_fqs_jiffies,
	.get = param_get_ulong,
};

static struct kernel_param_ops next_fqs_jiffies_ops = {
	.set = param_set_next_fqs_jiffies,
	.get = param_get_ulong,
};

module_param_cb(jiffies_till_first_fqs, &first_fqs_jiffies_ops, &jiffies_till_first_fqs, 0644);
module_param_cb(jiffies_till_next_fqs, &next_fqs_jiffies_ops, &jiffies_till_next_fqs, 0644);
module_param(rcu_kick_kthreads, bool, 0644);

static void force_qs_rnp(int (*f)(struct rcu_data *rdp));
static int rcu_pending(int user);

/*
 * Return the number of RCU GPs completed thus far for debug & stats.
 */
unsigned long rcu_get_gp_seq(void)
{
	return READ_ONCE(rcu_state_p->gp_seq);
}
EXPORT_SYMBOL_GPL(rcu_get_gp_seq);

/*
 * Return the number of RCU-sched GPs completed thus far for debug & stats.
 */
unsigned long rcu_sched_get_gp_seq(void)
{
	return READ_ONCE(rcu_sched_state.gp_seq);
}
EXPORT_SYMBOL_GPL(rcu_sched_get_gp_seq);

/*
 * Return the number of RCU-bh GPs completed thus far for debug & stats.
 */
unsigned long rcu_bh_get_gp_seq(void)
{
	return READ_ONCE(rcu_bh_state.gp_seq);
}
EXPORT_SYMBOL_GPL(rcu_bh_get_gp_seq);

/*
 * Return the number of RCU expedited batches completed thus far for
 * debug & stats.  Odd numbers mean that a batch is in progress, even
 * numbers mean idle.  The value returned will thus be roughly double
 * the cumulative batches since boot.
 */
unsigned long rcu_exp_batches_completed(void)
{
	return rcu_state_p->expedited_sequence;
}
EXPORT_SYMBOL_GPL(rcu_exp_batches_completed);

/*
 * Return the number of RCU-sched expedited batches completed thus far
 * for debug & stats.  Similar to rcu_exp_batches_completed().
 */
unsigned long rcu_exp_batches_completed_sched(void)
{
	return rcu_sched_state.expedited_sequence;
}
EXPORT_SYMBOL_GPL(rcu_exp_batches_completed_sched);

/*
 * Send along grace-period-related data for rcutorture diagnostics.
 */
void rcutorture_get_gp_data(enum rcutorture_type test_type, int *flags,
			    unsigned long *gp_seq)
{
	struct rcu_state *rsp = NULL;

	switch (test_type) {
	case RCU_FLAVOR:
		rsp = rcu_state_p;
		break;
	case RCU_BH_FLAVOR:
		rsp = &rcu_bh_state;
		break;
	case RCU_SCHED_FLAVOR:
		rsp = &rcu_sched_state;
		break;
	default:
		break;
	}
	if (rsp == NULL)
		return;
	*flags = READ_ONCE(rsp->gp_flags);
	*gp_seq = rcu_seq_current(&rsp->gp_seq);
}
EXPORT_SYMBOL_GPL(rcutorture_get_gp_data);

/*
 * Return the root node of the specified rcu_state structure.
 */
static struct rcu_node *rcu_get_root(struct rcu_state *rsp)
{
	return &rsp->node[0];
}

/*
 * Enter an RCU extended quiescent state, which can be either the
 * idle loop or adaptive-tickless usermode execution.
 *
 * We crowbar the ->dynticks_nmi_nesting field to zero to allow for
 * the possibility of usermode upcalls having messed up our count
 * of interrupt nesting level during the prior busy period.
 */
static noinstr void rcu_eqs_enter(bool user)
{
	struct rcu_state *rsp;
	struct rcu_data *rdp;
	struct rcu_dynticks *rdtp;

	rdtp = this_cpu_ptr(&rcu_dynticks);
	WRITE_ONCE(rdtp->dynticks_nmi_nesting, 0);
	WARN_ON_ONCE(IS_ENABLED(CONFIG_RCU_EQS_DEBUG) &&
		     rdp->dynticks_nesting == 0);
	if (rdp->dynticks_nesting != 1) {
		// RCU will still be watching, so just do accounting and leave.
		rdp->dynticks_nesting--;
		return;
	}

	lockdep_assert_irqs_disabled();
	instrumentation_begin();
	trace_rcu_dyntick(TPS("Start"), rdp->dynticks_nesting, 0, atomic_read(&rdp->dynticks));
	WARN_ON_ONCE(IS_ENABLED(CONFIG_RCU_EQS_DEBUG) && !user && !is_idle_task(current));
	for_each_rcu_flavor(rsp) {
		rdp = this_cpu_ptr(rsp->rda);
		do_nocb_deferred_wakeup(rdp);
	}
	rcu_prepare_for_idle();
	rcu_preempt_deferred_qs(current);

	// instrumentation for the noinstr rcu_dynticks_eqs_enter()
	instrument_atomic_write(&rdp->dynticks, sizeof(rdp->dynticks));

	instrumentation_end();
	WRITE_ONCE(rdp->dynticks_nesting, 0); /* Avoid irq-access tearing. */
	// RCU is watching here ...
	rcu_dynticks_eqs_enter();
	// ... but is no longer watching here.
	rcu_dynticks_task_enter();
}

/**
 * rcu_idle_enter - inform RCU that current CPU is entering idle
 *
 * Enter idle mode, in other words, -leave- the mode in which RCU
 * read-side critical sections can occur.  (Though RCU read-side
 * critical sections can occur in irq handlers in idle, a possibility
 * handled by irq_enter() and irq_exit().)
 *
 * If you add or remove a call to rcu_idle_enter(), be sure to test with
 * CONFIG_RCU_EQS_DEBUG=y.
 */
void rcu_idle_enter(void)
{
	lockdep_assert_irqs_disabled();
	rcu_eqs_enter(false);
}

#ifdef CONFIG_NO_HZ_FULL
/**
 * rcu_user_enter - inform RCU that we are resuming userspace.
 *
 * Enter RCU idle mode right before resuming userspace.  No use of RCU
 * is permitted between this call and rcu_user_exit(). This way the
 * CPU doesn't need to maintain the tick for RCU maintenance purposes
 * when the CPU runs in userspace.
 *
 * If you add or remove a call to rcu_user_enter(), be sure to test with
 * CONFIG_RCU_EQS_DEBUG=y.
 */
noinstr void rcu_user_enter(void)
{
	lockdep_assert_irqs_disabled();
	rcu_eqs_enter(true);
}
#endif /* CONFIG_NO_HZ_FULL */

/**
 * rcu_nmi_exit - inform RCU of exit from NMI context
 *
 * If we are returning from the outermost NMI handler that interrupted an
 * RCU-idle period, update rdp->dynticks and rdp->dynticks_nmi_nesting
 * to let the RCU grace-period handling know that the CPU is back to
 * being RCU-idle.
 *
 * If you add or remove a call to rcu_nmi_exit(), be sure to test
 * with CONFIG_RCU_EQS_DEBUG=y.
 */
noinstr void rcu_nmi_exit(void)
{
	struct rcu_data *rdp = this_cpu_ptr(&rcu_data);

	instrumentation_begin();
	/*
	 * Check for ->dynticks_nmi_nesting underflow and bad ->dynticks.
	 * (We are exiting an NMI handler, so RCU better be paying attention
	 * to us!)
	 */
	WARN_ON_ONCE(rdp->dynticks_nmi_nesting <= 0);
	WARN_ON_ONCE(rcu_dynticks_curr_cpu_in_eqs());

	/*
	 * If the nesting level is not 1, the CPU wasn't RCU-idle, so
	 * leave it in non-RCU-idle state.
	 */
	if (rdp->dynticks_nmi_nesting != 1) {
		trace_rcu_dyntick(TPS("--="), rdp->dynticks_nmi_nesting, rdp->dynticks_nmi_nesting - 2,
				  atomic_read(&rdp->dynticks));
		WRITE_ONCE(rdp->dynticks_nmi_nesting, /* No store tearing. */
			   rdp->dynticks_nmi_nesting - 2);
		instrumentation_end();
		return;
	}

	/* This NMI interrupted an RCU-idle CPU, restore RCU-idleness. */
	trace_rcu_dyntick(TPS("Startirq"), rdp->dynticks_nmi_nesting, 0, atomic_read(&rdp->dynticks));
	WRITE_ONCE(rdp->dynticks_nmi_nesting, 0); /* Avoid store tearing. */

	if (!in_nmi())
		rcu_prepare_for_idle();

	// instrumentation for the noinstr rcu_dynticks_eqs_enter()
	instrument_atomic_write(&rdp->dynticks, sizeof(rdp->dynticks));
	instrumentation_end();

	// RCU is watching here ...
	rcu_dynticks_eqs_enter();
	// ... but is no longer watching here.

	if (!in_nmi())
		rcu_dynticks_task_enter();
}

/**
 * rcu_irq_exit - inform RCU that current CPU is exiting irq towards idle
 *
 * Exit from an interrupt handler, which might possibly result in entering
 * idle mode, in other words, leaving the mode in which read-side critical
 * sections can occur.  The caller must have disabled interrupts.
 *
 * This code assumes that the idle loop never does anything that might
 * result in unbalanced calls to irq_enter() and irq_exit().  If your
 * architecture's idle loop violates this assumption, RCU will give you what
 * you deserve, good and hard.  But very infrequently and irreproducibly.
 *
 * Use things like work queues to work around this limitation.
 *
 * You have been warned.
 *
 * If you add or remove a call to rcu_irq_exit(), be sure to test with
 * CONFIG_RCU_EQS_DEBUG=y.
 */
void noinstr rcu_irq_exit(void)
{
	struct rcu_dynticks *rdtp = this_cpu_ptr(&rcu_dynticks);

	lockdep_assert_irqs_disabled();
	rcu_nmi_exit();
}

/**
 * rcu_irq_exit_preempt - Inform RCU that current CPU is exiting irq
 *			  towards in kernel preemption
 *
 * Same as rcu_irq_exit() but has a sanity check that scheduling is safe
 * from RCU point of view. Invoked from return from interrupt before kernel
 * preemption.
 */
void rcu_irq_exit_preempt(void)
{
	lockdep_assert_irqs_disabled();
	rcu_nmi_exit();

	RCU_LOCKDEP_WARN(__this_cpu_read(rcu_data.dynticks_nesting) <= 0,
			 "RCU dynticks_nesting counter underflow/zero!");
	RCU_LOCKDEP_WARN(__this_cpu_read(rcu_data.dynticks_nmi_nesting) !=
			 DYNTICK_IRQ_NONIDLE,
			 "Bad RCU  dynticks_nmi_nesting counter\n");
	RCU_LOCKDEP_WARN(rcu_dynticks_curr_cpu_in_eqs(),
			 "RCU in extended quiescent state!");
}

#ifdef CONFIG_PROVE_RCU
/**
 * rcu_irq_exit_check_preempt - Validate that scheduling is possible
 */
void rcu_irq_exit_check_preempt(void)
{
	lockdep_assert_irqs_disabled();

	RCU_LOCKDEP_WARN(__this_cpu_read(rcu_data.dynticks_nesting) <= 0,
			 "RCU dynticks_nesting counter underflow/zero!");
	RCU_LOCKDEP_WARN(__this_cpu_read(rcu_data.dynticks_nmi_nesting) !=
			 DYNTICK_IRQ_NONIDLE,
			 "Bad RCU  dynticks_nmi_nesting counter\n");
	RCU_LOCKDEP_WARN(rcu_dynticks_curr_cpu_in_eqs(),
			 "RCU in extended quiescent state!");
}
#endif /* #ifdef CONFIG_PROVE_RCU */

/*
 * Wrapper for rcu_irq_exit() where interrupts are enabled.
 *
 * If you add or remove a call to rcu_irq_exit_irqson(), be sure to test
 * with CONFIG_RCU_EQS_DEBUG=y.
 */
void rcu_irq_exit_irqson(void)
{
	unsigned long flags;

	local_irq_save(flags);
	rcu_irq_exit();
	local_irq_restore(flags);
}

/*
 * Exit an RCU extended quiescent state, which can be either the
 * idle loop or adaptive-tickless usermode execution.
 *
 * We crowbar the ->dynticks_nmi_nesting field to DYNTICK_IRQ_NONIDLE to
 * allow for the possibility of usermode upcalls messing up our count of
 * interrupt nesting level during the busy period that is just now starting.
 */
static void noinstr rcu_eqs_exit(bool user)
{
	struct rcu_dynticks *rdtp;
	long oldval;

	lockdep_assert_irqs_disabled();
	rdtp = this_cpu_ptr(&rcu_dynticks);
	oldval = rdtp->dynticks_nesting;
	WARN_ON_ONCE(IS_ENABLED(CONFIG_RCU_EQS_DEBUG) && oldval < 0);
	if (oldval) {
		rdtp->dynticks_nesting++;
		return;
	}
	rcu_dynticks_task_exit();
	// RCU is not watching here ...
	rcu_dynticks_eqs_exit();
	// ... but is watching here.
	instrumentation_begin();

	// instrumentation for the noinstr rcu_dynticks_eqs_exit()
	instrument_atomic_write(&rdp->dynticks, sizeof(rdp->dynticks));

	rcu_cleanup_after_idle();
	trace_rcu_dyntick(TPS("End"), rdtp->dynticks_nesting, 1, rdtp->dynticks);
	WARN_ON_ONCE(IS_ENABLED(CONFIG_RCU_EQS_DEBUG) && !user && !is_idle_task(current));
	WRITE_ONCE(rdp->dynticks_nesting, 1);
	WARN_ON_ONCE(rdp->dynticks_nmi_nesting);
	WRITE_ONCE(rdp->dynticks_nmi_nesting, DYNTICK_IRQ_NONIDLE);
	instrumentation_end();
}

/**
 * rcu_idle_exit - inform RCU that current CPU is leaving idle
 *
 * Exit idle mode, in other words, -enter- the mode in which RCU
 * read-side critical sections can occur.
 *
 * If you add or remove a call to rcu_idle_exit(), be sure to test with
 * CONFIG_RCU_EQS_DEBUG=y.
 */
void rcu_idle_exit(void)
{
	unsigned long flags;

	local_irq_save(flags);
	rcu_eqs_exit(false);
	local_irq_restore(flags);
}

#ifdef CONFIG_NO_HZ_FULL
/**
 * rcu_user_exit - inform RCU that we are exiting userspace.
 *
 * Exit RCU idle mode while entering the kernel because it can
 * run a RCU read side critical section anytime.
 *
 * If you add or remove a call to rcu_user_exit(), be sure to test with
 * CONFIG_RCU_EQS_DEBUG=y.
 */
void noinstr rcu_user_exit(void)
{
	rcu_eqs_exit(1);
}

/**
 * __rcu_irq_enter_check_tick - Enable scheduler tick on CPU if RCU needs it.
 *
 * The scheduler tick is not normally enabled when CPUs enter the kernel
 * from nohz_full userspace execution.  After all, nohz_full userspace
 * execution is an RCU quiescent state and the time executing in the kernel
 * is quite short.  Except of course when it isn't.  And it is not hard to
 * cause a large system to spend tens of seconds or even minutes looping
 * in the kernel, which can cause a number of problems, include RCU CPU
 * stall warnings.
 *
 * Therefore, if a nohz_full CPU fails to report a quiescent state
 * in a timely manner, the RCU grace-period kthread sets that CPU's
 * ->rcu_urgent_qs flag with the expectation that the next interrupt or
 * exception will invoke this function, which will turn on the scheduler
 * tick, which will enable RCU to detect that CPU's quiescent states,
 * for example, due to cond_resched() calls in CONFIG_PREEMPT=n kernels.
 * The tick will be disabled once a quiescent state is reported for
 * this CPU.
 *
 * Of course, in carefully tuned systems, there might never be an
 * interrupt or exception.  In that case, the RCU grace-period kthread
 * will eventually cause one to happen.  However, in less carefully
 * controlled environments, this function allows RCU to get what it
 * needs without creating otherwise useless interruptions.
 */
void __rcu_irq_enter_check_tick(void)
{
	struct rcu_data *rdp = this_cpu_ptr(&rcu_data);

	 // Enabling the tick is unsafe in NMI handlers.
	if (WARN_ON_ONCE(in_nmi()))
		return;

	RCU_LOCKDEP_WARN(rcu_dynticks_curr_cpu_in_eqs(),
			 "Illegal rcu_irq_enter_check_tick() from extended quiescent state");

	if (!tick_nohz_full_cpu(rdp->cpu) ||
	    !READ_ONCE(rdp->rcu_urgent_qs) ||
	    READ_ONCE(rdp->rcu_forced_tick)) {
		// RCU doesn't need nohz_full help from this CPU, or it is
		// already getting that help.
		return;
	}

	// We get here only when not in an extended quiescent state and
	// from interrupts (as opposed to NMIs).  Therefore, (1) RCU is
	// already watching and (2) The fact that we are in an interrupt
	// handler and that the rcu_node lock is an irq-disabled lock
	// prevents self-deadlock.  So we can safely recheck under the lock.
	// Note that the nohz_full state currently cannot change.
	raw_spin_lock_rcu_node(rdp->mynode);
	if (rdp->rcu_urgent_qs && !rdp->rcu_forced_tick) {
		// A nohz_full CPU is in the kernel and RCU needs a
		// quiescent state.  Turn on the tick!
		WRITE_ONCE(rdp->rcu_forced_tick, true);
		tick_dep_set_cpu(rdp->cpu, TICK_DEP_BIT_RCU);
	}
	raw_spin_unlock_rcu_node(rdp->mynode);
}
#endif /* CONFIG_NO_HZ_FULL */

/**
 * rcu_nmi_enter - inform RCU of entry to NMI context
 *
 * If the CPU was idle from RCU's viewpoint, update rdtp->dynticks and
 * rdtp->dynticks_nmi_nesting to let the RCU grace-period handling know
 * that the CPU is active.  This implementation permits nested NMIs, as
 * long as the nesting level does not overflow an int.  (You will probably
 * run out of stack space first.)
 *
 * If you add or remove a call to rcu_nmi_enter(), be sure to test
 * with CONFIG_RCU_EQS_DEBUG=y.
 */
noinstr void rcu_nmi_enter(void)
{
	long incby = 2;
	struct rcu_data *rdp = this_cpu_ptr(&rcu_data);

	/* Complain about underflow. */
	WARN_ON_ONCE(rdtp->dynticks_nmi_nesting < 0);

	/*
	 * If idle from RCU viewpoint, atomically increment ->dynticks
	 * to mark non-idle and increment ->dynticks_nmi_nesting by one.
	 * Otherwise, increment ->dynticks_nmi_nesting by two.  This means
	 * if ->dynticks_nmi_nesting is equal to one, we are guaranteed
	 * to be in the outermost NMI handler that interrupted an RCU-idle
	 * period (observation due to Andy Lutomirski).
	 */
	if (rcu_dynticks_curr_cpu_in_eqs()) {

		if (!in_nmi())
			rcu_dynticks_task_exit();

		// RCU is not watching here ...
		rcu_dynticks_eqs_exit();
		// ... but is watching here.

		if (!in_nmi()) {
			instrumentation_begin();
			rcu_cleanup_after_idle();
			instrumentation_end();
		}

		instrumentation_begin();
		// instrumentation for the noinstr rcu_dynticks_curr_cpu_in_eqs()
		instrument_atomic_read(&rdp->dynticks, sizeof(rdp->dynticks));
		// instrumentation for the noinstr rcu_dynticks_eqs_exit()
		instrument_atomic_write(&rdp->dynticks, sizeof(rdp->dynticks));

		incby = 1;
	} else if (!in_nmi()) {
		instrumentation_begin();
		rcu_irq_enter_check_tick();
		instrumentation_end();
	} else  {
		instrumentation_begin();
	}

	trace_rcu_dyntick(incby == 1 ? TPS("Endirq") : TPS("++="),
			  rdp->dynticks_nmi_nesting,
			  rdp->dynticks_nmi_nesting + incby, atomic_read(&rdp->dynticks));
	instrumentation_end();
	WRITE_ONCE(rdp->dynticks_nmi_nesting, /* Prevent store tearing. */
		   rdp->dynticks_nmi_nesting + incby);
	barrier();
}

/**
 * rcu_irq_enter - inform RCU that current CPU is entering irq away from idle
 *
 * Enter an interrupt handler, which might possibly result in exiting
 * idle mode, in other words, entering the mode in which read-side critical
 * sections can occur.  The caller must have disabled interrupts.
 *
 * Note that the Linux kernel is fully capable of entering an interrupt
 * handler that it never exits, for example when doing upcalls to user mode!
 * This code assumes that the idle loop never does upcalls to user mode.
 * If your architecture's idle loop does do upcalls to user mode (or does
 * anything else that results in unbalanced calls to the irq_enter() and
 * irq_exit() functions), RCU will give you what you deserve, good and hard.
 * But very infrequently and irreproducibly.
 *
 * Use things like work queues to work around this limitation.
 *
 * You have been warned.
 *
 * If you add or remove a call to rcu_irq_enter(), be sure to test with
 * CONFIG_RCU_EQS_DEBUG=y.
 */
noinstr void rcu_irq_enter(void)
{
	struct rcu_dynticks *rdtp = this_cpu_ptr(&rcu_dynticks);

	lockdep_assert_irqs_disabled();
	rcu_nmi_enter();
}

/*
 * Wrapper for rcu_irq_enter() where interrupts are enabled.
 *
 * If you add or remove a call to rcu_irq_enter_irqson(), be sure to test
 * with CONFIG_RCU_EQS_DEBUG=y.
 */
void rcu_irq_enter_irqson(void)
{
	unsigned long flags;

	local_irq_save(flags);
	rcu_irq_enter();
	local_irq_restore(flags);
}

/*
 * If any sort of urgency was applied to the current CPU (for example,
 * the scheduler-clock interrupt was enabled on a nohz_full CPU) in order
 * to get to a quiescent state, disable it.
 */
static void rcu_disable_urgency_upon_qs(struct rcu_data *rdp)
{
	raw_lockdep_assert_held_rcu_node(rdp->mynode);
	WRITE_ONCE(rdp->rcu_urgent_qs, false);
	WRITE_ONCE(rdp->rcu_need_heavy_qs, false);
	if (tick_nohz_full_cpu(rdp->cpu) && rdp->rcu_forced_tick) {
		tick_dep_clear_cpu(rdp->cpu, TICK_DEP_BIT_RCU);
		WRITE_ONCE(rdp->rcu_forced_tick, false);
	}
}

noinstr bool __rcu_is_watching(void)
{
	return !rcu_dynticks_curr_cpu_in_eqs();
}

/**
 * rcu_is_watching - see if RCU thinks that the current CPU is idle
 *
 * Return true if RCU is watching the running CPU, which means that this
 * CPU can safely enter RCU read-side critical sections.  In other words,
 * if the current CPU is in its idle loop and is neither in an interrupt
 * or NMI handler, return true.
 */
bool rcu_is_watching(void)
{
	bool ret;

	preempt_disable_notrace();
	ret = !rcu_dynticks_curr_cpu_in_eqs();
	preempt_enable_notrace();
	return ret;
}
EXPORT_SYMBOL_GPL(rcu_is_watching);

/*
 * If a holdout task is actually running, request an urgent quiescent
 * state from its CPU.  This is unsynchronized, so migrations can cause
 * the request to go to the wrong CPU.  Which is OK, all that will happen
 * is that the CPU's next context switch will be a bit slower and next
 * time around this task will generate another request.
 */
void rcu_request_urgent_qs_task(struct task_struct *t)
{
	int cpu;

	barrier();
	cpu = task_cpu(t);
	if (!task_curr(t))
		return; /* This task is not running on that CPU. */
	smp_store_release(per_cpu_ptr(&rcu_dynticks.rcu_urgent_qs, cpu), true);
}

#if defined(CONFIG_PROVE_RCU) && defined(CONFIG_HOTPLUG_CPU)

/*
 * Is the current CPU online as far as RCU is concerned?
 *
 * Disable preemption to avoid false positives that could otherwise
 * happen due to the current CPU number being sampled, this task being
 * preempted, its old CPU being taken offline, resuming on some other CPU,
 * then determining that its old CPU is now offline.  Because there are
 * multiple flavors of RCU, and because this function can be called in the
 * midst of updating the flavors while a given CPU coming online or going
 * offline, it is necessary to check all flavors.  If any of the flavors
 * believe that given CPU is online, it is considered to be online.
 *
 * Disable checking if in an NMI handler because we cannot safely
 * report errors from NMI handlers anyway.  In addition, it is OK to use
 * RCU on an offline processor during initial boot, hence the check for
 * rcu_scheduler_fully_active.
 */
bool rcu_lockdep_current_cpu_online(void)
{
	struct rcu_data *rdp;
	struct rcu_node *rnp;
	struct rcu_state *rsp;

	if (in_nmi() || !rcu_scheduler_fully_active)
		return true;
	preempt_disable_notrace();
	rdp = this_cpu_ptr(&rcu_data);
	rnp = rdp->mynode;
	if (rdp->grpmask & rcu_rnp_online_cpus(rnp))
		ret = true;
	preempt_enable_notrace();
	return ret;
}
EXPORT_SYMBOL_GPL(rcu_lockdep_current_cpu_online);

#endif /* #if defined(CONFIG_PROVE_RCU) && defined(CONFIG_HOTPLUG_CPU) */

/**
 * rcu_is_cpu_rrupt_from_idle - see if idle or immediately interrupted from idle
 *
 * If the current CPU is idle or running at a first-level (not nested)
 * interrupt from idle, return true.  The caller must have at least
 * disabled preemption.
 */
static int rcu_is_cpu_rrupt_from_idle(void)
{
	return __this_cpu_read(rcu_dynticks.dynticks_nesting) <= 0 &&
	       __this_cpu_read(rcu_dynticks.dynticks_nmi_nesting) <= 1;
}

/*
 * We are reporting a quiescent state on behalf of some other CPU, so
 * it is our responsibility to check for and handle potential overflow
 * of the rcu_node ->gp_seq counter with respect to the rcu_data counters.
 * After all, the CPU might be in deep idle state, and thus executing no
 * code whatsoever.
 */
static void rcu_gpnum_ovf(struct rcu_node *rnp, struct rcu_data *rdp)
{
	raw_lockdep_assert_held_rcu_node(rnp);
	if (ULONG_CMP_LT(rcu_seq_current(&rdp->gp_seq) + ULONG_MAX / 4,
			 rnp->gp_seq))
		WRITE_ONCE(rdp->gpwrap, true);
	if (ULONG_CMP_LT(rdp->rcu_iw_gp_seq + ULONG_MAX / 4, rnp->gp_seq))
		rdp->rcu_iw_gp_seq = rnp->gp_seq + ULONG_MAX / 4;
}

/*
 * Snapshot the specified CPU's dynticks counter so that we can later
 * credit them with an implicit quiescent state.  Return 1 if this CPU
 * is in dynticks idle mode, which is an extended quiescent state.
 */
static int dyntick_save_progress_counter(struct rcu_data *rdp)
{
	rdp->dynticks_snap = rcu_dynticks_snap(rdp->dynticks);
	if (rcu_dynticks_in_eqs(rdp->dynticks_snap)) {
		trace_rcu_fqs(rdp->rsp->name, rdp->gp_seq, rdp->cpu, TPS("dti"));
		rcu_gpnum_ovf(rdp->mynode, rdp);
		return 1;
	}
	return 0;
}

/*
 * Handler for the irq_work request posted when a grace period has
 * gone on for too long, but not yet long enough for an RCU CPU
 * stall warning.  Set state appropriately, but just complain if
 * there is unexpected state on entry.
 */
static void rcu_iw_handler(struct irq_work *iwp)
{
	struct rcu_data *rdp;
	struct rcu_node *rnp;

	rdp = container_of(iwp, struct rcu_data, rcu_iw);
	rnp = rdp->mynode;
	raw_spin_lock_rcu_node(rnp);
	if (!WARN_ON_ONCE(!rdp->rcu_iw_pending)) {
		rdp->rcu_iw_gp_seq = rnp->gp_seq;
		rdp->rcu_iw_pending = false;
	}
	raw_spin_unlock_rcu_node(rnp);
}

/*
 * Return true if the specified CPU has passed through a quiescent
 * state by virtue of being in or having passed through an dynticks
 * idle state since the last call to dyntick_save_progress_counter()
 * for this same CPU, or by virtue of having been offline.
 */
static int rcu_implicit_dynticks_qs(struct rcu_data *rdp)
{
	unsigned long jtsq;
	bool *rnhqp;
	bool *ruqp;
	struct rcu_node *rnp = rdp->mynode;

	/*
	 * If the CPU passed through or entered a dynticks idle phase with
	 * no active irq/NMI handlers, then we can safely pretend that the CPU
	 * already acknowledged the request to pass through a quiescent
	 * state.  Either way, that CPU cannot possibly be in an RCU
	 * read-side critical section that started before the beginning
	 * of the current RCU grace period.
	 */
	if (rcu_dynticks_in_eqs_since(rdp->dynticks, rdp->dynticks_snap)) {
		trace_rcu_fqs(rdp->rsp->name, rdp->gp_seq, rdp->cpu, TPS("dti"));
		rdp->dynticks_fqs++;
		rcu_gpnum_ovf(rnp, rdp);
		return 1;
	}

	/*
	 * Has this CPU encountered a cond_resched() since the beginning
	 * of the grace period?  For this to be the case, the CPU has to
	 * have noticed the current grace period.  This might not be the
	 * case for nohz_full CPUs looping in the kernel.
	 */
	jtsq = jiffies_till_sched_qs;
	ruqp = per_cpu_ptr(&rcu_dynticks.rcu_urgent_qs, rdp->cpu);
	if (time_after(jiffies, rdp->rsp->gp_start + jtsq) &&
	    READ_ONCE(rdp->rcu_qs_ctr_snap) != per_cpu(rcu_dynticks.rcu_qs_ctr, rdp->cpu) &&
	    rcu_seq_current(&rdp->gp_seq) == rnp->gp_seq && !rdp->gpwrap) {
		trace_rcu_fqs(rdp->rsp->name, rdp->gp_seq, rdp->cpu, TPS("rqc"));
		rcu_gpnum_ovf(rnp, rdp);
		return 1;
	} else if (time_after(jiffies, rdp->rsp->gp_start + jtsq)) {
		/* Load rcu_qs_ctr before store to rcu_urgent_qs. */
		smp_store_release(ruqp, true);
	}

	/* If waiting too long on an offline CPU, complain. */
	if (!(rdp->grpmask & rcu_rnp_online_cpus(rnp)) &&
	    time_after(jiffies, rdp->rsp->gp_start + HZ)) {
		bool onl;
		struct rcu_node *rnp1;

		WARN_ON(1);  /* Offline CPUs are supposed to report QS! */
		pr_info("%s: grp: %d-%d level: %d ->gp_seq %ld ->completedqs %ld\n",
			__func__, rnp->grplo, rnp->grphi, rnp->level,
			(long)rnp->gp_seq, (long)rnp->completedqs);
		for (rnp1 = rnp; rnp1; rnp1 = rnp1->parent)
			pr_info("%s: %d:%d ->qsmask %#lx ->qsmaskinit %#lx ->qsmaskinitnext %#lx ->rcu_gp_init_mask %#lx\n",
				__func__, rnp1->grplo, rnp1->grphi, rnp1->qsmask, rnp1->qsmaskinit, rnp1->qsmaskinitnext, rnp1->rcu_gp_init_mask);
		onl = !!(rdp->grpmask & rcu_rnp_online_cpus(rnp));
		pr_info("%s %d: %c online: %ld(%d) offline: %ld(%d)\n",
			__func__, rdp->cpu, ".o"[onl],
			(long)rdp->rcu_onl_gp_seq, rdp->rcu_onl_gp_flags,
			(long)rdp->rcu_ofl_gp_seq, rdp->rcu_ofl_gp_flags);
		return 1; /* Break things loose after complaining. */
	}

	/*
	 * A CPU running for an extended time within the kernel can
	 * delay RCU grace periods.  When the CPU is in NO_HZ_FULL mode,
	 * even context-switching back and forth between a pair of
	 * in-kernel CPU-bound tasks cannot advance grace periods.
	 * So if the grace period is old enough, make the CPU pay attention.
	 * Note that the unsynchronized assignments to the per-CPU
	 * rcu_need_heavy_qs variable are safe.  Yes, setting of
	 * bits can be lost, but they will be set again on the next
	 * force-quiescent-state pass.  So lost bit sets do not result
	 * in incorrect behavior, merely in a grace period lasting
	 * a few jiffies longer than it might otherwise.  Because
	 * there are at most four threads involved, and because the
	 * updates are only once every few jiffies, the probability of
	 * lossage (and thus of slight grace-period extension) is
	 * quite low.
	 */
	rnhqp = &per_cpu(rcu_dynticks.rcu_need_heavy_qs, rdp->cpu);
	if (!READ_ONCE(*rnhqp) &&
	    (time_after(jiffies, rcu_state.gp_start + jtsq * 2) ||
	     time_after(jiffies, rcu_state.jiffies_resched) ||
	     rcu_state.cbovld)) {
		WRITE_ONCE(*rnhqp, true);
		/* Store rcu_need_heavy_qs before rcu_urgent_qs. */
		smp_store_release(ruqp, true);
		rdp->rsp->jiffies_resched += jtsq; /* Re-enable beating. */
	}

	/*
	 * If more than halfway to RCU CPU stall-warning time, do a
	 * resched_cpu() to try to loosen things up a bit.  Also check to
	 * see if the CPU is getting hammered with interrupts, but only
	 * once per grace period, just to keep the IPIs down to a dull roar.
	 */
	if (tick_nohz_full_cpu(rdp->cpu) &&
	    (time_after(jiffies, READ_ONCE(rdp->last_fqs_resched) + jtsq * 3) ||
	     rcu_state.cbovld)) {
		WRITE_ONCE(*ruqp, true);
		resched_cpu(rdp->cpu);
		if (IS_ENABLED(CONFIG_IRQ_WORK) &&
		    !rdp->rcu_iw_pending && rdp->rcu_iw_gp_seq != rnp->gp_seq &&
		    (rnp->ffmask & rdp->grpmask)) {
			init_irq_work(&rdp->rcu_iw, rcu_iw_handler);
			atomic_set(&rdp->rcu_iw.flags, IRQ_WORK_HARD_IRQ);
			rdp->rcu_iw_pending = true;
			rdp->rcu_iw_gp_seq = rnp->gp_seq;
			irq_work_queue_on(&rdp->rcu_iw, rdp->cpu);
		}
	}

	return 0;
}

static void record_gp_stall_check_time(struct rcu_state *rsp)
{
	unsigned long j = jiffies;
	unsigned long j1;

	rsp->gp_start = j;
	j1 = rcu_jiffies_till_stall_check();
	/* Record ->gp_start before ->jiffies_stall. */
	smp_store_release(&rsp->jiffies_stall, j + j1); /* ^^^ */
	rsp->jiffies_resched = j + j1 / 2;
	rsp->n_force_qs_gpstart = READ_ONCE(rsp->n_force_qs);
}

/*
 * Convert a ->gp_state value to a character string.
 */
static const char *gp_state_getname(short gs)
{
	if (gs < 0 || gs >= ARRAY_SIZE(gp_state_names))
		return "???";
	return gp_state_names[gs];
}

/*
 * Complain about starvation of grace-period kthread.
 */
static void rcu_check_gp_kthread_starvation(struct rcu_state *rsp)
{
	unsigned long gpa;
	unsigned long j;

	j = jiffies;
	gpa = READ_ONCE(rsp->gp_activity);
	if (j - gpa > 2 * HZ) {
		pr_err("%s kthread starved for %ld jiffies! g%ld f%#x %s(%d) ->state=%#lx ->cpu=%d\n",
		       rsp->name, j - gpa,
		       (long)rcu_seq_current(&rsp->gp_seq),
		       rsp->gp_flags,
		       gp_state_getname(rsp->gp_state), rsp->gp_state,
		       rsp->gp_kthread ? rsp->gp_kthread->state : ~0,
		       rsp->gp_kthread ? task_cpu(rsp->gp_kthread) : -1);
		if (rsp->gp_kthread) {
			pr_err("RCU grace-period kthread stack dump:\n");
			sched_show_task(rsp->gp_kthread);
			wake_up_process(rsp->gp_kthread);
		}
	}
}

/*
 * Dump stacks of all tasks running on stalled CPUs.  First try using
 * NMIs, but fall back to manual remote stack tracing on architectures
 * that don't support NMI-based stack dumps.  The NMI-triggered stack
 * traces are more accurate because they are printed by the target CPU.
 */
static void rcu_dump_cpu_stacks(struct rcu_state *rsp)
{
	int cpu;
	unsigned long flags;
	struct rcu_node *rnp;

	rcu_for_each_leaf_node(rsp, rnp) {
		raw_spin_lock_irqsave_rcu_node(rnp, flags);
		for_each_leaf_node_possible_cpu(rnp, cpu)
			if (rnp->qsmask & leaf_node_cpu_bit(rnp, cpu))
				if (!trigger_single_cpu_backtrace(cpu))
					dump_cpu_task(cpu);
		raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
	}
}

/*
 * If too much time has passed in the current grace period, and if
 * so configured, go kick the relevant kthreads.
 */
static void rcu_stall_kick_kthreads(struct rcu_state *rsp)
{
	unsigned long j;

	if (!rcu_kick_kthreads)
		return;
	j = READ_ONCE(rsp->jiffies_kick_kthreads);
	if (time_after(jiffies, j) && rsp->gp_kthread &&
	    (rcu_gp_in_progress(rsp) || READ_ONCE(rsp->gp_flags))) {
		WARN_ONCE(1, "Kicking %s grace-period kthread\n", rsp->name);
		rcu_ftrace_dump(DUMP_ALL);
		wake_up_process(rsp->gp_kthread);
		WRITE_ONCE(rsp->jiffies_kick_kthreads, j + HZ);
	}
}

static void panic_on_rcu_stall(void)
{
	if (sysctl_panic_on_rcu_stall)
		panic("RCU Stall\n");
}

static void print_other_cpu_stall(struct rcu_state *rsp, unsigned long gp_seq)
{
	int cpu;
	unsigned long flags;
	unsigned long gpa;
	unsigned long j;
	int ndetected = 0;
	struct rcu_node *rnp = rcu_get_root(rsp);
	long totqlen = 0;

	/* Kick and suppress, if so configured. */
	rcu_stall_kick_kthreads(rsp);
	if (rcu_cpu_stall_suppress)
		return;

	/*
	 * OK, time to rat on our buddy...
	 * See Documentation/RCU/stallwarn.txt for info on how to debug
	 * RCU CPU stall warnings.
	 */
	pr_err("INFO: %s detected stalls on CPUs/tasks:", rsp->name);
	print_cpu_stall_info_begin();
	rcu_for_each_leaf_node(rsp, rnp) {
		raw_spin_lock_irqsave_rcu_node(rnp, flags);
		ndetected += rcu_print_task_stall(rnp);
		if (rnp->qsmask != 0) {
			for_each_leaf_node_possible_cpu(rnp, cpu)
				if (rnp->qsmask & leaf_node_cpu_bit(rnp, cpu)) {
					print_cpu_stall_info(rsp, cpu);
					ndetected++;
				}
		}
		raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
	}

	print_cpu_stall_info_end();
	for_each_possible_cpu(cpu)
		totqlen += rcu_segcblist_n_cbs(&per_cpu_ptr(rsp->rda,
							    cpu)->cblist);
	pr_cont("(detected by %d, t=%ld jiffies, g=%ld, q=%lu)\n",
	       smp_processor_id(), (long)(jiffies - rsp->gp_start),
	       (long)rcu_seq_current(&rsp->gp_seq), totqlen);
	if (ndetected) {
		rcu_dump_cpu_stacks(rsp);

		/* Complain about tasks blocking the grace period. */
		rcu_print_detail_task_stall(rsp);
	} else {
		if (rcu_seq_current(&rsp->gp_seq) != gp_seq) {
			pr_err("INFO: Stall ended before state dump start\n");
		} else {
			j = jiffies;
			gpa = READ_ONCE(rsp->gp_activity);
			pr_err("All QSes seen, last %s kthread activity %ld (%ld-%ld), jiffies_till_next_fqs=%ld, root ->qsmask %#lx\n",
			       rsp->name, j - gpa, j, gpa,
			       jiffies_till_next_fqs,
			       rcu_get_root(rsp)->qsmask);
			/* In this case, the current CPU might be at fault. */
			sched_show_task(current);
		}
	}
	/* Rewrite if needed in case of slow consoles. */
	if (ULONG_CMP_GE(jiffies, READ_ONCE(rsp->jiffies_stall)))
		WRITE_ONCE(rsp->jiffies_stall,
			   jiffies + 3 * rcu_jiffies_till_stall_check() + 3);

	rcu_check_gp_kthread_starvation(rsp);

	panic_on_rcu_stall();

	force_quiescent_state(rsp);  /* Kick them all. */
}

static void print_cpu_stall(struct rcu_state *rsp)
{
	int cpu;
	unsigned long flags;
	struct rcu_data *rdp = this_cpu_ptr(rsp->rda);
	struct rcu_node *rnp = rcu_get_root(rsp);
	long totqlen = 0;

	/* Kick and suppress, if so configured. */
	rcu_stall_kick_kthreads(rsp);
	if (rcu_cpu_stall_suppress)
		return;

	/*
	 * OK, time to rat on ourselves...
	 * See Documentation/RCU/stallwarn.txt for info on how to debug
	 * RCU CPU stall warnings.
	 */
	pr_err("INFO: %s self-detected stall on CPU", rsp->name);
	print_cpu_stall_info_begin();
	raw_spin_lock_irqsave_rcu_node(rdp->mynode, flags);
	print_cpu_stall_info(rsp, smp_processor_id());
	raw_spin_unlock_irqrestore_rcu_node(rdp->mynode, flags);
	print_cpu_stall_info_end();
	for_each_possible_cpu(cpu)
		totqlen += rcu_segcblist_n_cbs(&per_cpu_ptr(rsp->rda,
							    cpu)->cblist);
	pr_cont(" (t=%lu jiffies g=%ld q=%lu)\n",
		jiffies - rsp->gp_start,
		(long)rcu_seq_current(&rsp->gp_seq), totqlen);

	rcu_check_gp_kthread_starvation(rsp);

	rcu_dump_cpu_stacks(rsp);

	raw_spin_lock_irqsave_rcu_node(rnp, flags);
	/* Rewrite if needed in case of slow consoles. */
	if (ULONG_CMP_GE(jiffies, READ_ONCE(rsp->jiffies_stall)))
		WRITE_ONCE(rsp->jiffies_stall,
			   jiffies + 3 * rcu_jiffies_till_stall_check() + 3);
	raw_spin_unlock_irqrestore_rcu_node(rnp, flags);

	panic_on_rcu_stall();

	/*
	 * Attempt to revive the RCU machinery by forcing a context switch.
	 *
	 * A context switch would normally allow the RCU state machine to make
	 * progress and it could be we're stuck in kernel space without context
	 * switches for an entirely unreasonable amount of time.
	 */
	resched_cpu(smp_processor_id());
}

static void check_cpu_stall(struct rcu_state *rsp, struct rcu_data *rdp)
{
	unsigned long gs1;
	unsigned long gs2;
	unsigned long gps;
	unsigned long j;
	unsigned long jn;
	unsigned long js;
	struct rcu_node *rnp;

	if ((rcu_cpu_stall_suppress && !rcu_kick_kthreads) ||
	    !rcu_gp_in_progress(rsp))
		return;
	rcu_stall_kick_kthreads(rsp);
	j = jiffies;

	/*
	 * Lots of memory barriers to reject false positives.
	 *
	 * The idea is to pick up rsp->gp_seq, then rsp->jiffies_stall,
	 * then rsp->gp_start, and finally another copy of rsp->gp_seq.
	 * These values are updated in the opposite order with memory
	 * barriers (or equivalent) during grace-period initialization
	 * and cleanup.  Now, a false positive can occur if we get an new
	 * value of rsp->gp_start and a old value of rsp->jiffies_stall.
	 * But given the memory barriers, the only way that this can happen
	 * is if one grace period ends and another starts between these
	 * two fetches.  This is detected by comparing the second fetch
	 * of rsp->gp_seq with the previous fetch from rsp->gp_seq.
	 *
	 * Given this check, comparisons of jiffies, rsp->jiffies_stall,
	 * and rsp->gp_start suffice to forestall false positives.
	 */
	gs1 = READ_ONCE(rsp->gp_seq);
	smp_rmb(); /* Pick up ->gp_seq first... */
	js = READ_ONCE(rsp->jiffies_stall);
	smp_rmb(); /* ...then ->jiffies_stall before the rest... */
	gps = READ_ONCE(rsp->gp_start);
	smp_rmb(); /* ...and finally ->gp_start before ->gp_seq again. */
	gs2 = READ_ONCE(rsp->gp_seq);
	if (gs1 != gs2 ||
	    ULONG_CMP_LT(j, js) ||
	    ULONG_CMP_GE(gps, js))
		return; /* No stall or GP completed since entering function. */
	rnp = rdp->mynode;
	jn = jiffies + 3 * rcu_jiffies_till_stall_check() + 3;
	if (rcu_gp_in_progress(rsp) &&
	    (READ_ONCE(rnp->qsmask) & rdp->grpmask) &&
	    cmpxchg(&rsp->jiffies_stall, js, jn) == js) {

		/* We haven't checked in, so go dump stack. */
		print_cpu_stall(rsp);

	} else if (rcu_gp_in_progress(rsp) &&
		   ULONG_CMP_GE(j, js + RCU_STALL_RAT_DELAY) &&
		   cmpxchg(&rsp->jiffies_stall, js, jn) == js) {

		/* They had a few time units to dump stack, so complain. */
		print_other_cpu_stall(rsp, gs2);
	}
}

/**
 * rcu_cpu_stall_reset - prevent further stall warnings in current grace period
 *
 * Set the stall-warning timeout way off into the future, thus preventing
 * any RCU CPU stall-warning messages from appearing in the current set of
 * RCU grace periods.
 *
 * The caller must disable hard irqs.
 */
void rcu_cpu_stall_reset(void)
{
	struct rcu_state *rsp;

	for_each_rcu_flavor(rsp)
		WRITE_ONCE(rsp->jiffies_stall, jiffies + ULONG_MAX / 2);
}

/* Trace-event wrapper function for trace_rcu_future_grace_period.  */
static void trace_rcu_this_gp(struct rcu_node *rnp, struct rcu_data *rdp,
			      unsigned long gp_seq_req, const char *s)
{
	trace_rcu_future_grace_period(rcu_state.name, READ_ONCE(rnp->gp_seq),
				      gp_seq_req, rnp->level,
				      rnp->grplo, rnp->grphi, s);
}

/*
 * rcu_start_this_gp - Request the start of a particular grace period
 * @rnp_start: The leaf node of the CPU from which to start.
 * @rdp: The rcu_data corresponding to the CPU from which to start.
 * @gp_seq_req: The gp_seq of the grace period to start.
 *
 * Start the specified grace period, as needed to handle newly arrived
 * callbacks.  The required future grace periods are recorded in each
 * rcu_node structure's ->gp_seq_needed field.  Returns true if there
 * is reason to awaken the grace-period kthread.
 *
 * The caller must hold the specified rcu_node structure's ->lock, which
 * is why the caller is responsible for waking the grace-period kthread.
 *
 * Returns true if the GP thread needs to be awakened else false.
 */
static bool rcu_start_this_gp(struct rcu_node *rnp_start, struct rcu_data *rdp,
			      unsigned long gp_seq_req)
{
	bool ret = false;
	struct rcu_state *rsp = rdp->rsp;
	struct rcu_node *rnp;

	/*
	 * Use funnel locking to either acquire the root rcu_node
	 * structure's lock or bail out if the need for this grace period
	 * has already been recorded -- or if that grace period has in
	 * fact already started.  If there is already a grace period in
	 * progress in a non-leaf node, no recording is needed because the
	 * end of the grace period will scan the leaf rcu_node structures.
	 * Note that rnp_start->lock must not be released.
	 */
	raw_lockdep_assert_held_rcu_node(rnp_start);
	trace_rcu_this_gp(rnp_start, rdp, gp_seq_req, TPS("Startleaf"));
	for (rnp = rnp_start; 1; rnp = rnp->parent) {
		if (rnp != rnp_start)
			raw_spin_lock_rcu_node(rnp);
		if (ULONG_CMP_GE(rnp->gp_seq_needed, gp_seq_req) ||
		    rcu_seq_started(&rnp->gp_seq, gp_seq_req) ||
		    (rnp != rnp_start &&
		     rcu_seq_state(rcu_seq_current(&rnp->gp_seq)))) {
			trace_rcu_this_gp(rnp, rdp, gp_seq_req,
					  TPS("Prestarted"));
			goto unlock_out;
		}
		WRITE_ONCE(rnp->gp_seq_needed, gp_seq_req);
		if (rcu_seq_state(rcu_seq_current(&rnp->gp_seq))) {
			/*
			 * We just marked the leaf or internal node, and a
			 * grace period is in progress, which means that
			 * rcu_gp_cleanup() will see the marking.  Bail to
			 * reduce contention.
			 */
			trace_rcu_this_gp(rnp_start, rdp, gp_seq_req,
					  TPS("Startedleaf"));
			goto unlock_out;
		}
		if (rnp != rnp_start && rnp->parent != NULL)
			raw_spin_unlock_rcu_node(rnp);
		if (!rnp->parent)
			break;  /* At root, and perhaps also leaf. */
	}

	/* If GP already in progress, just leave, otherwise start one. */
	if (rcu_gp_in_progress(rsp)) {
		trace_rcu_this_gp(rnp, rdp, gp_seq_req, TPS("Startedleafroot"));
		goto unlock_out;
	}
	trace_rcu_this_gp(rnp, rdp, gp_seq_req, TPS("Startedroot"));
	WRITE_ONCE(rcu_state.gp_flags, rcu_state.gp_flags | RCU_GP_FLAG_INIT);
	WRITE_ONCE(rcu_state.gp_req_activity, jiffies);
	if (!READ_ONCE(rcu_state.gp_kthread)) {
		trace_rcu_this_gp(rnp, rdp, gp_seq_req, TPS("NoGPkthread"));
		goto unlock_out;
	}
	trace_rcu_grace_period(rcu_state.name, data_race(rcu_state.gp_seq), TPS("newreq"));
	ret = true;  /* Caller must wake GP kthread. */
unlock_out:
	/* Push furthest requested GP to leaf node and rcu_data structure. */
	if (ULONG_CMP_LT(gp_seq_req, rnp->gp_seq_needed)) {
		WRITE_ONCE(rnp_start->gp_seq_needed, rnp->gp_seq_needed);
		WRITE_ONCE(rdp->gp_seq_needed, rnp->gp_seq_needed);
	}
	if (rnp != rnp_start)
		raw_spin_unlock_rcu_node(rnp);
	return ret;
}

/*
 * Clean up any old requests for the just-ended grace period.  Also return
 * whether any additional grace periods have been requested.
 */
static bool rcu_future_gp_cleanup(struct rcu_state *rsp, struct rcu_node *rnp)
{
	bool needmore;
	struct rcu_data *rdp = this_cpu_ptr(rsp->rda);

	needmore = ULONG_CMP_LT(rnp->gp_seq, rnp->gp_seq_needed);
	if (!needmore)
		rnp->gp_seq_needed = rnp->gp_seq; /* Avoid counter wrap. */
	trace_rcu_this_gp(rnp, rdp, rnp->gp_seq,
			  needmore ? TPS("CleanupMore") : TPS("Cleanup"));
	return needmore;
}

/*
 * Awaken the grace-period kthread.  Don't do a self-awaken (unless in
 * an interrupt or softirq handler), and don't bother awakening when there
 * is nothing for the grace-period kthread to do (as in several CPUs raced
 * to awaken, and we lost), and finally don't try to awaken a kthread that
 * has not yet been created.  If all those checks are passed, track some
 * debug information and awaken.
 *
 * So why do the self-wakeup when in an interrupt or softirq handler
 * in the grace-period kthread's context?  Because the kthread might have
 * been interrupted just as it was going to sleep, and just after the final
 * pre-sleep check of the awaken condition.  In this case, a wakeup really
 * is required, and is therefore supplied.
 */
static void rcu_gp_kthread_wake(struct rcu_state *rsp)
{
	struct task_struct *t = READ_ONCE(rcu_state.gp_kthread);

	if ((current == t && !in_irq() && !in_serving_softirq()) ||
	    !READ_ONCE(rcu_state.gp_flags) || !t)
		return;
	swake_up_one(&rsp->gp_wq);
}

/*
 * If there is room, assign a ->gp_seq number to any callbacks on this
 * CPU that have not already been assigned.  Also accelerate any callbacks
 * that were previously assigned a ->gp_seq number that has since proven
 * to be too conservative, which can happen if callbacks get assigned a
 * ->gp_seq number while RCU is idle, but with reference to a non-root
 * rcu_node structure.  This function is idempotent, so it does not hurt
 * to call it repeatedly.  Returns an flag saying that we should awaken
 * the RCU grace-period kthread.
 *
 * The caller must hold rnp->lock with interrupts disabled.
 */
static bool rcu_accelerate_cbs(struct rcu_state *rsp, struct rcu_node *rnp,
			       struct rcu_data *rdp)
{
	unsigned long gp_seq_req;
	bool ret = false;

	raw_lockdep_assert_held_rcu_node(rnp);

	/* If no pending (not yet ready to invoke) callbacks, nothing to do. */
	if (!rcu_segcblist_pend_cbs(&rdp->cblist))
		return false;

	/*
	 * Callbacks are often registered with incomplete grace-period
	 * information.  Something about the fact that getting exact
	 * information requires acquiring a global lock...  RCU therefore
	 * makes a conservative estimate of the grace period number at which
	 * a given callback will become ready to invoke.	The following
	 * code checks this estimate and improves it when possible, thus
	 * accelerating callback invocation to an earlier grace-period
	 * number.
	 */
	gp_seq_req = rcu_seq_snap(&rsp->gp_seq);
	if (rcu_segcblist_accelerate(&rdp->cblist, gp_seq_req))
		ret = rcu_start_this_gp(rnp, rdp, gp_seq_req);

	/* Trace depending on how much we were able to accelerate. */
	if (rcu_segcblist_restempty(&rdp->cblist, RCU_WAIT_TAIL))
		trace_rcu_grace_period(rsp->name, rdp->gp_seq, TPS("AccWaitCB"));
	else
		trace_rcu_grace_period(rsp->name, rdp->gp_seq, TPS("AccReadyCB"));
	return ret;
}

/*
 * Similar to rcu_accelerate_cbs(), but does not require that the leaf
 * rcu_node structure's ->lock be held.  It consults the cached value
 * of ->gp_seq_needed in the rcu_data structure, and if that indicates
 * that a new grace-period request be made, invokes rcu_accelerate_cbs()
 * while holding the leaf rcu_node structure's ->lock.
 */
static void rcu_accelerate_cbs_unlocked(struct rcu_state *rsp,
					struct rcu_node *rnp,
					struct rcu_data *rdp)
{
	unsigned long c;
	bool needwake;

	rcu_lockdep_assert_cblist_protected(rdp);
	c = rcu_seq_snap(&rcu_state.gp_seq);
	if (!READ_ONCE(rdp->gpwrap) && ULONG_CMP_GE(rdp->gp_seq_needed, c)) {
		/* Old request still live, so mark recent callbacks. */
		(void)rcu_segcblist_accelerate(&rdp->cblist, c);
		return;
	}
	raw_spin_lock_rcu_node(rnp); /* irqs already disabled. */
	needwake = rcu_accelerate_cbs(rsp, rnp, rdp);
	raw_spin_unlock_rcu_node(rnp); /* irqs remain disabled. */
	if (needwake)
		rcu_gp_kthread_wake(rsp);
}

/*
 * Move any callbacks whose grace period has completed to the
 * RCU_DONE_TAIL sublist, then compact the remaining sublists and
 * assign ->gp_seq numbers to any callbacks in the RCU_NEXT_TAIL
 * sublist.  This function is idempotent, so it does not hurt to
 * invoke it repeatedly.  As long as it is not invoked -too- often...
 * Returns true if the RCU grace-period kthread needs to be awakened.
 *
 * The caller must hold rnp->lock with interrupts disabled.
 */
static bool rcu_advance_cbs(struct rcu_state *rsp, struct rcu_node *rnp,
			    struct rcu_data *rdp)
{
	raw_lockdep_assert_held_rcu_node(rnp);

	/* If no pending (not yet ready to invoke) callbacks, nothing to do. */
	if (!rcu_segcblist_pend_cbs(&rdp->cblist))
		return false;

	/*
	 * Find all callbacks whose ->gp_seq numbers indicate that they
	 * are ready to invoke, and put them into the RCU_DONE_TAIL sublist.
	 */
	rcu_segcblist_advance(&rdp->cblist, rnp->gp_seq);

	/* Classify any remaining callbacks. */
	return rcu_accelerate_cbs(rsp, rnp, rdp);
}

/*
 * Update CPU-local rcu_data state to record the beginnings and ends of
 * grace periods.  The caller must hold the ->lock of the leaf rcu_node
 * structure corresponding to the current CPU, and must have irqs disabled.
 * Returns true if the grace-period kthread needs to be awakened.
 */
static bool __note_gp_changes(struct rcu_state *rsp, struct rcu_node *rnp,
			      struct rcu_data *rdp)
{
	bool ret = false;
	bool need_qs;
	const bool offloaded = IS_ENABLED(CONFIG_RCU_NOCB_CPU) &&
			       rcu_segcblist_is_offloaded(&rdp->cblist);

	raw_lockdep_assert_held_rcu_node(rnp);

	if (rdp->gp_seq == rnp->gp_seq)
		return false; /* Nothing to do. */

	/* Handle the ends of any preceding grace periods first. */
	if (rcu_seq_completed_gp(rdp->gp_seq, rnp->gp_seq) ||
	    unlikely(READ_ONCE(rdp->gpwrap))) {
		if (!offloaded)
			ret = rcu_advance_cbs(rnp, rdp); /* Advance CBs. */
		rdp->core_needs_qs = false;
		trace_rcu_grace_period(rcu_state.name, rdp->gp_seq, TPS("cpuend"));
	} else {
		if (!offloaded)
			ret = rcu_accelerate_cbs(rnp, rdp); /* Recent CBs. */
		if (rdp->core_needs_qs)
			rdp->core_needs_qs = !!(rnp->qsmask & rdp->grpmask);
	}

	/* Now handle the beginnings of any new-to-this-CPU grace periods. */
	if (rcu_seq_new_gp(rdp->gp_seq, rnp->gp_seq) ||
	    unlikely(READ_ONCE(rdp->gpwrap))) {
		/*
		 * If the current grace period is waiting for this CPU,
		 * set up to detect a quiescent state, otherwise don't
		 * go looking for one.
		 */
		trace_rcu_grace_period(rcu_state.name, rnp->gp_seq, TPS("cpustart"));
		need_qs = !!(rnp->qsmask & rdp->grpmask);
		rdp->cpu_no_qs.b.norm = need_qs;
		rdp->core_needs_qs = need_qs;
		zero_cpu_stall_ticks(rdp);
	}
	rdp->gp_seq = rnp->gp_seq;  /* Remember new grace-period state. */
	if (ULONG_CMP_LT(rdp->gp_seq_needed, rnp->gp_seq_needed) || rdp->gpwrap)
		WRITE_ONCE(rdp->gp_seq_needed, rnp->gp_seq_needed);
	WRITE_ONCE(rdp->gpwrap, false);
	rcu_gpnum_ovf(rnp, rdp);
	return ret;
}

static void note_gp_changes(struct rcu_state *rsp, struct rcu_data *rdp)
{
	unsigned long flags;
	bool needwake;
	struct rcu_node *rnp;

	local_irq_save(flags);
	rnp = rdp->mynode;
	if ((rdp->gp_seq == rcu_seq_current(&rnp->gp_seq) &&
	     !unlikely(READ_ONCE(rdp->gpwrap))) || /* w/out lock. */
	    !raw_spin_trylock_rcu_node(rnp)) { /* irqs already off, so later. */
		local_irq_restore(flags);
		return;
	}
	needwake = __note_gp_changes(rsp, rnp, rdp);
	raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
	if (needwake)
		rcu_gp_kthread_wake(rsp);
}

static void rcu_gp_slow(struct rcu_state *rsp, int delay)
{
	if (delay > 0 &&
	    !(rcu_seq_ctr(rsp->gp_seq) %
	      (rcu_num_nodes * PER_RCU_NODE_PERIOD * delay)))
		schedule_timeout_idle(delay);
}

static unsigned long sleep_duration;

/* Allow rcutorture to stall the grace-period kthread. */
void rcu_gp_set_torture_wait(int duration)
{
	if (IS_ENABLED(CONFIG_RCU_TORTURE_TEST) && duration > 0)
		WRITE_ONCE(sleep_duration, duration);
}
EXPORT_SYMBOL_GPL(rcu_gp_set_torture_wait);

/* Actually implement the aforementioned wait. */
static void rcu_gp_torture_wait(void)
{
	unsigned long duration;

	if (!IS_ENABLED(CONFIG_RCU_TORTURE_TEST))
		return;
	duration = xchg(&sleep_duration, 0UL);
	if (duration > 0) {
		pr_alert("%s: Waiting %lu jiffies\n", __func__, duration);
		schedule_timeout_idle(duration);
		pr_alert("%s: Wait complete\n", __func__);
	}
}

/*
 * Initialize a new grace period.  Return false if no grace period required.
 */
static bool rcu_gp_init(struct rcu_state *rsp)
{
	unsigned long flags;
	unsigned long oldmask;
	unsigned long mask;
	struct rcu_data *rdp;
	struct rcu_node *rnp = rcu_get_root(rsp);

	WRITE_ONCE(rsp->gp_activity, jiffies);
	raw_spin_lock_irq_rcu_node(rnp);
	if (!READ_ONCE(rsp->gp_flags)) {
		/* Spurious wakeup, tell caller to go back to sleep.  */
		raw_spin_unlock_irq_rcu_node(rnp);
		return false;
	}
	WRITE_ONCE(rsp->gp_flags, 0); /* Clear all flags: New grace period. */

	if (WARN_ON_ONCE(rcu_gp_in_progress(rsp))) {
		/*
		 * Grace period already in progress, don't start another.
		 * Not supposed to be able to happen.
		 */
		raw_spin_unlock_irq_rcu_node(rnp);
		return false;
	}

	/* Advance to a new grace period and initialize state. */
	record_gp_stall_check_time(rsp);
	/* Record GP times before starting GP, hence rcu_seq_start(). */
	rcu_seq_start(&rcu_state.gp_seq);
	ASSERT_EXCLUSIVE_WRITER(rcu_state.gp_seq);
	trace_rcu_grace_period(rcu_state.name, rcu_state.gp_seq, TPS("start"));
	raw_spin_unlock_irq_rcu_node(rnp);

	/*
	 * Apply per-leaf buffered online and offline operations to the
	 * rcu_node tree.  Note that this new grace period need not wait
	 * for subsequent online CPUs, and that quiescent-state forcing
	 * will handle subsequent offline CPUs.
	 */
	rsp->gp_state = RCU_GP_ONOFF;
	rcu_for_each_leaf_node(rsp, rnp) {
		spin_lock(&rsp->ofl_lock);
		raw_spin_lock_irq_rcu_node(rnp);
		if (rnp->qsmaskinit == rnp->qsmaskinitnext &&
		    !rnp->wait_blkd_tasks) {
			/* Nothing to do on this leaf rcu_node structure. */
			raw_spin_unlock_irq_rcu_node(rnp);
			spin_unlock(&rsp->ofl_lock);
			continue;
		}

		/* Record old state, apply changes to ->qsmaskinit field. */
		oldmask = rnp->qsmaskinit;
		rnp->qsmaskinit = rnp->qsmaskinitnext;

		/* If zero-ness of ->qsmaskinit changed, propagate up tree. */
		if (!oldmask != !rnp->qsmaskinit) {
			if (!oldmask) { /* First online CPU for rcu_node. */
				if (!rnp->wait_blkd_tasks) /* Ever offline? */
					rcu_init_new_rnp(rnp);
			} else if (rcu_preempt_has_tasks(rnp)) {
				rnp->wait_blkd_tasks = true; /* blocked tasks */
			} else { /* Last offline CPU and can propagate. */
				rcu_cleanup_dead_rnp(rnp);
			}
		}

		/*
		 * If all waited-on tasks from prior grace period are
		 * done, and if all this rcu_node structure's CPUs are
		 * still offline, propagate up the rcu_node tree and
		 * clear ->wait_blkd_tasks.  Otherwise, if one of this
		 * rcu_node structure's CPUs has since come back online,
		 * simply clear ->wait_blkd_tasks.
		 */
		if (rnp->wait_blkd_tasks &&
		    (!rcu_preempt_has_tasks(rnp) || rnp->qsmaskinit)) {
			rnp->wait_blkd_tasks = false;
			if (!rnp->qsmaskinit)
				rcu_cleanup_dead_rnp(rnp);
		}

		raw_spin_unlock_irq_rcu_node(rnp);
		spin_unlock(&rsp->ofl_lock);
	}
	rcu_gp_slow(rsp, gp_preinit_delay); /* Races with CPU hotplug. */

	/*
	 * Set the quiescent-state-needed bits in all the rcu_node
	 * structures for all currently online CPUs in breadth-first order,
	 * starting from the root rcu_node structure, relying on the layout
	 * of the tree within the rsp->node[] array.  Note that other CPUs
	 * will access only the leaves of the hierarchy, thus seeing that no
	 * grace period is in progress, at least until the corresponding
	 * leaf node has been initialized.
	 *
	 * The grace period cannot complete until the initialization
	 * process finishes, because this kthread handles both.
	 */
	rsp->gp_state = RCU_GP_INIT;
	rcu_for_each_node_breadth_first(rsp, rnp) {
		rcu_gp_slow(rsp, gp_init_delay);
		raw_spin_lock_irqsave_rcu_node(rnp, flags);
		rdp = this_cpu_ptr(rsp->rda);
		rcu_preempt_check_blocked_tasks(rsp, rnp);
		rnp->qsmask = rnp->qsmaskinit;
		WRITE_ONCE(rnp->gp_seq, rsp->gp_seq);
		if (rnp == rdp->mynode)
			(void)__note_gp_changes(rsp, rnp, rdp);
		rcu_preempt_boost_start_gp(rnp);
		trace_rcu_grace_period_init(rsp->name, rnp->gp_seq,
					    rnp->level, rnp->grplo,
					    rnp->grphi, rnp->qsmask);
		/* Quiescent states for tasks on any now-offline CPUs. */
		mask = rnp->qsmask & ~rnp->qsmaskinitnext;
		rnp->rcu_gp_init_mask = mask;
		if ((mask || rnp->wait_blkd_tasks) && rcu_is_leaf_node(rnp))
			rcu_report_qs_rnp(mask, rsp, rnp, rnp->gp_seq, flags);
		else
			raw_spin_unlock_irq_rcu_node(rnp);
		cond_resched_tasks_rcu_qs();
		WRITE_ONCE(rsp->gp_activity, jiffies);
	}

	return true;
}

/*
 * Helper function for swait_event_idle_exclusive() wakeup at force-quiescent-state
 * time.
 */
static bool rcu_gp_fqs_check_wake(struct rcu_state *rsp, int *gfp)
{
	struct rcu_node *rnp = rcu_get_root(rsp);

	// If under overload conditions, force an immediate FQS scan.
	if (*gfp & RCU_GP_FLAG_OVLD)
		return true;

	// Someone like call_rcu() requested a force-quiescent-state scan.
	*gfp = READ_ONCE(rcu_state.gp_flags);
	if (*gfp & RCU_GP_FLAG_FQS)
		return true;

	// The current grace period has completed.
	if (!READ_ONCE(rnp->qsmask) && !rcu_preempt_blocked_readers_cgp(rnp))
		return true;

	return false;
}

/*
 * Do one round of quiescent-state forcing.
 */
static void rcu_gp_fqs(struct rcu_state *rsp, bool first_time)
{
	struct rcu_node *rnp = rcu_get_root(rsp);

	WRITE_ONCE(rsp->gp_activity, jiffies);
	rsp->n_force_qs++;
	if (first_time) {
		/* Collect dyntick-idle snapshots. */
		force_qs_rnp(rsp, dyntick_save_progress_counter);
	} else {
		/* Handle dyntick-idle and offline CPUs. */
		force_qs_rnp(rsp, rcu_implicit_dynticks_qs);
	}
	/* Clear flag to prevent immediate re-entry. */
	if (READ_ONCE(rsp->gp_flags) & RCU_GP_FLAG_FQS) {
		raw_spin_lock_irq_rcu_node(rnp);
		WRITE_ONCE(rsp->gp_flags,
			   READ_ONCE(rsp->gp_flags) & ~RCU_GP_FLAG_FQS);
		raw_spin_unlock_irq_rcu_node(rnp);
	}
}

/*
 * Loop doing repeated quiescent-state forcing until the grace period ends.
 */
static void rcu_gp_fqs_loop(void)
{
	bool first_gp_fqs;
	int gf = 0;
	unsigned long j;
	int ret;
	struct rcu_node *rnp = rcu_get_root();

	first_gp_fqs = true;
	j = READ_ONCE(jiffies_till_first_fqs);
	if (rcu_state.cbovld)
		gf = RCU_GP_FLAG_OVLD;
	ret = 0;
	for (;;) {
		if (!ret) {
			rcu_state.jiffies_force_qs = jiffies + j;
			WRITE_ONCE(rcu_state.jiffies_kick_kthreads,
				   jiffies + (j ? 3 * j : 2));
		}
		trace_rcu_grace_period(rcu_state.name, rcu_state.gp_seq,
				       TPS("fqswait"));
		rcu_state.gp_state = RCU_GP_WAIT_FQS;
		ret = swait_event_idle_timeout_exclusive(
				rcu_state.gp_wq, rcu_gp_fqs_check_wake(&gf), j);
		rcu_gp_torture_wait();
		rcu_state.gp_state = RCU_GP_DOING_FQS;
		/* Locking provides needed memory barriers. */
		/* If grace period done, leave loop. */
		if (!READ_ONCE(rnp->qsmask) &&
		    !rcu_preempt_blocked_readers_cgp(rnp))
			break;
		/* If time for quiescent-state forcing, do it. */
		if (!time_after(rcu_state.jiffies_force_qs, jiffies) ||
		    (gf & RCU_GP_FLAG_FQS)) {
			trace_rcu_grace_period(rcu_state.name, rcu_state.gp_seq,
					       TPS("fqsstart"));
			rcu_gp_fqs(first_gp_fqs);
			gf = 0;
			if (first_gp_fqs) {
				first_gp_fqs = false;
				gf = rcu_state.cbovld ? RCU_GP_FLAG_OVLD : 0;
			}
			trace_rcu_grace_period(rcu_state.name, rcu_state.gp_seq,
					       TPS("fqsend"));
			cond_resched_tasks_rcu_qs();
			WRITE_ONCE(rcu_state.gp_activity, jiffies);
			ret = 0; /* Force full wait till next FQS. */
			j = READ_ONCE(jiffies_till_next_fqs);
		} else {
			/* Deal with stray signal. */
			cond_resched_tasks_rcu_qs();
			WRITE_ONCE(rcu_state.gp_activity, jiffies);
			WARN_ON(signal_pending(current));
			trace_rcu_grace_period(rcu_state.name, rcu_state.gp_seq,
					       TPS("fqswaitsig"));
			ret = 1; /* Keep old FQS timing. */
			j = jiffies;
			if (time_after(jiffies, rcu_state.jiffies_force_qs))
				j = 1;
			else
				j = rcu_state.jiffies_force_qs - j;
			gf = 0;
		}
	}
}

/*
 * Clean up after the old grace period.
 */
static void rcu_gp_cleanup(struct rcu_state *rsp)
{
	int cpu;
	bool needgp = false;
	unsigned long gp_duration;
	unsigned long new_gp_seq;
	struct rcu_data *rdp;
	struct rcu_node *rnp = rcu_get_root(rsp);
	struct swait_queue_head *sq;

	WRITE_ONCE(rsp->gp_activity, jiffies);
	raw_spin_lock_irq_rcu_node(rnp);
	gp_duration = jiffies - rsp->gp_start;
	if (gp_duration > rsp->gp_max)
		rsp->gp_max = gp_duration;

	/*
	 * We know the grace period is complete, but to everyone else
	 * it appears to still be ongoing.  But it is also the case
	 * that to everyone else it looks like there is nothing that
	 * they can do to advance the grace period.  It is therefore
	 * safe for us to drop the lock in order to mark the grace
	 * period as completed in all of the rcu_node structures.
	 */
	raw_spin_unlock_irq_rcu_node(rnp);

	/*
	 * Propagate new ->gp_seq value to rcu_node structures so that
	 * other CPUs don't have to wait until the start of the next grace
	 * period to process their callbacks.  This also avoids some nasty
	 * RCU grace-period initialization races by forcing the end of
	 * the current grace period to be completely recorded in all of
	 * the rcu_node structures before the beginning of the next grace
	 * period is recorded in any of the rcu_node structures.
	 */
	new_gp_seq = rsp->gp_seq;
	rcu_seq_end(&new_gp_seq);
	rcu_for_each_node_breadth_first(rsp, rnp) {
		raw_spin_lock_irq_rcu_node(rnp);
		if (WARN_ON_ONCE(rcu_preempt_blocked_readers_cgp(rnp)))
			dump_blkd_tasks(rsp, rnp, 10);
		WARN_ON_ONCE(rnp->qsmask);
		WRITE_ONCE(rnp->gp_seq, new_gp_seq);
		rdp = this_cpu_ptr(rsp->rda);
		if (rnp == rdp->mynode)
			needgp = __note_gp_changes(rsp, rnp, rdp) || needgp;
		/* smp_mb() provided by prior unlock-lock pair. */
		needgp = rcu_future_gp_cleanup(rnp) || needgp;
		// Reset overload indication for CPUs no longer overloaded
		if (rcu_is_leaf_node(rnp))
			for_each_leaf_node_cpu_mask(rnp, cpu, rnp->cbovldmask) {
				rdp = per_cpu_ptr(&rcu_data, cpu);
				check_cb_ovld_locked(rdp, rnp);
			}
		sq = rcu_nocb_gp_get(rnp);
		raw_spin_unlock_irq_rcu_node(rnp);
		rcu_nocb_gp_cleanup(sq);
		cond_resched_tasks_rcu_qs();
		WRITE_ONCE(rsp->gp_activity, jiffies);
		rcu_gp_slow(rsp, gp_cleanup_delay);
	}
	rnp = rcu_get_root(rsp);
	raw_spin_lock_irq_rcu_node(rnp); /* GP before rsp->gp_seq update. */

	/* Declare grace period done, trace first to use old GP number. */
	trace_rcu_grace_period(rcu_state.name, rcu_state.gp_seq, TPS("end"));
	rcu_seq_end(&rcu_state.gp_seq);
	ASSERT_EXCLUSIVE_WRITER(rcu_state.gp_seq);
	rcu_state.gp_state = RCU_GP_IDLE;
	/* Check for GP requests since above loop. */
	rdp = this_cpu_ptr(rsp->rda);
	if (!needgp && ULONG_CMP_LT(rnp->gp_seq, rnp->gp_seq_needed)) {
		trace_rcu_this_gp(rnp, rdp, rnp->gp_seq_needed,
				  TPS("CleanupMore"));
		needgp = true;
	}
	/* Advance CBs to reduce false positives below. */
	offloaded = IS_ENABLED(CONFIG_RCU_NOCB_CPU) &&
		    rcu_segcblist_is_offloaded(&rdp->cblist);
	if ((offloaded || !rcu_accelerate_cbs(rnp, rdp)) && needgp) {
		WRITE_ONCE(rcu_state.gp_flags, RCU_GP_FLAG_INIT);
		WRITE_ONCE(rcu_state.gp_req_activity, jiffies);
		trace_rcu_grace_period(rcu_state.name,
				       rcu_state.gp_seq,
				       TPS("newreq"));
	} else {
		WRITE_ONCE(rsp->gp_flags, rsp->gp_flags & RCU_GP_FLAG_INIT);
	}
	raw_spin_unlock_irq_rcu_node(rnp);
}

/*
 * Body of kthread that handles grace periods.
 */
static int __noreturn rcu_gp_kthread(void *arg)
{
	bool first_gp_fqs;
	int gf;
	unsigned long j;
	int ret;
	struct rcu_state *rsp = arg;
	struct rcu_node *rnp = rcu_get_root(rsp);

	rcu_bind_gp_kthread();
	for (;;) {

		/* Handle grace-period start. */
		for (;;) {
			trace_rcu_grace_period(rcu_state.name, rcu_state.gp_seq,
					       TPS("reqwait"));
			rcu_state.gp_state = RCU_GP_WAIT_GPS;
			swait_event_idle_exclusive(rcu_state.gp_wq,
					 READ_ONCE(rcu_state.gp_flags) &
					 RCU_GP_FLAG_INIT);
			rcu_gp_torture_wait();
			rcu_state.gp_state = RCU_GP_DONE_GPS;
			/* Locking provides needed memory barrier. */
			if (rcu_gp_init(rsp))
				break;
			cond_resched_tasks_rcu_qs();
			WRITE_ONCE(rsp->gp_activity, jiffies);
			WARN_ON(signal_pending(current));
			trace_rcu_grace_period(rcu_state.name, rcu_state.gp_seq,
					       TPS("reqwaitsig"));
		}

		/* Handle quiescent-state forcing. */
		first_gp_fqs = true;
		j = jiffies_till_first_fqs;
		ret = 0;
		for (;;) {
			if (!ret) {
				rsp->jiffies_force_qs = jiffies + j;
				WRITE_ONCE(rsp->jiffies_kick_kthreads,
					   jiffies + 3 * j);
			}
			trace_rcu_grace_period(rsp->name,
					       READ_ONCE(rsp->gp_seq),
					       TPS("fqswait"));
			rsp->gp_state = RCU_GP_WAIT_FQS;
			ret = swait_event_idle_timeout_exclusive(rsp->gp_wq,
					rcu_gp_fqs_check_wake(rsp, &gf), j);
			rsp->gp_state = RCU_GP_DOING_FQS;
			/* Locking provides needed memory barriers. */
			/* If grace period done, leave loop. */
			if (!READ_ONCE(rnp->qsmask) &&
			    !rcu_preempt_blocked_readers_cgp(rnp))
				break;
			/* If time for quiescent-state forcing, do it. */
			if (ULONG_CMP_GE(jiffies, rsp->jiffies_force_qs) ||
			    (gf & RCU_GP_FLAG_FQS)) {
				trace_rcu_grace_period(rsp->name,
						       READ_ONCE(rsp->gp_seq),
						       TPS("fqsstart"));
				rcu_gp_fqs(rsp, first_gp_fqs);
				first_gp_fqs = false;
				trace_rcu_grace_period(rsp->name,
						       READ_ONCE(rsp->gp_seq),
						       TPS("fqsend"));
				cond_resched_tasks_rcu_qs();
				WRITE_ONCE(rsp->gp_activity, jiffies);
				ret = 0; /* Force full wait till next FQS. */
				j = jiffies_till_next_fqs;
			} else {
				/* Deal with stray signal. */
				cond_resched_tasks_rcu_qs();
				WRITE_ONCE(rsp->gp_activity, jiffies);
				WARN_ON(signal_pending(current));
				trace_rcu_grace_period(rsp->name,
						       READ_ONCE(rsp->gp_seq),
						       TPS("fqswaitsig"));
				ret = 1; /* Keep old FQS timing. */
				j = jiffies;
				if (time_after(jiffies, rsp->jiffies_force_qs))
					j = 1;
				else
					j = rsp->jiffies_force_qs - j;
			}
		}

		/* Handle grace-period end. */
		rsp->gp_state = RCU_GP_CLEANUP;
		rcu_gp_cleanup(rsp);
		rsp->gp_state = RCU_GP_CLEANED;
	}
}

/*
 * Report a full set of quiescent states to the specified rcu_state data
 * structure.  Invoke rcu_gp_kthread_wake() to awaken the grace-period
 * kthread if another grace period is required.  Whether we wake
 * the grace-period kthread or it awakens itself for the next round
 * of quiescent-state forcing, that kthread will clean up after the
 * just-completed grace period.  Note that the caller must hold rnp->lock,
 * which is released before return.
 */
static void rcu_report_qs_rsp(struct rcu_state *rsp, unsigned long flags)
	__releases(rcu_get_root(rsp)->lock)
{
	raw_lockdep_assert_held_rcu_node(rcu_get_root(rsp));
	WARN_ON_ONCE(!rcu_gp_in_progress(rsp));
	WRITE_ONCE(rsp->gp_flags, READ_ONCE(rsp->gp_flags) | RCU_GP_FLAG_FQS);
	raw_spin_unlock_irqrestore_rcu_node(rcu_get_root(rsp), flags);
	rcu_gp_kthread_wake(rsp);
}

/*
 * Similar to rcu_report_qs_rdp(), for which it is a helper function.
 * Allows quiescent states for a group of CPUs to be reported at one go
 * to the specified rcu_node structure, though all the CPUs in the group
 * must be represented by the same rcu_node structure (which need not be a
 * leaf rcu_node structure, though it often will be).  The gps parameter
 * is the grace-period snapshot, which means that the quiescent states
 * are valid only if rnp->gp_seq is equal to gps.  That structure's lock
 * must be held upon entry, and it is released before return.
 *
 * As a special case, if mask is zero, the bit-already-cleared check is
 * disabled.  This allows propagating quiescent state due to resumed tasks
 * during grace-period initialization.
 */
static void
rcu_report_qs_rnp(unsigned long mask, struct rcu_state *rsp,
		  struct rcu_node *rnp, unsigned long gps, unsigned long flags)
	__releases(rnp->lock)
{
	unsigned long oldmask = 0;
	struct rcu_node *rnp_c;

	raw_lockdep_assert_held_rcu_node(rnp);

	/* Walk up the rcu_node hierarchy. */
	for (;;) {
		if ((!(rnp->qsmask & mask) && mask) || rnp->gp_seq != gps) {

			/*
			 * Our bit has already been cleared, or the
			 * relevant grace period is already over, so done.
			 */
			raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
			return;
		}
		WARN_ON_ONCE(oldmask); /* Any child must be all zeroed! */
		WARN_ON_ONCE(!rcu_is_leaf_node(rnp) &&
			     rcu_preempt_blocked_readers_cgp(rnp));
		WRITE_ONCE(rnp->qsmask, rnp->qsmask & ~mask);
		trace_rcu_quiescent_state_report(rcu_state.name, rnp->gp_seq,
						 mask, rnp->qsmask, rnp->level,
						 rnp->grplo, rnp->grphi,
						 !!rnp->gp_tasks);
		if (rnp->qsmask != 0 || rcu_preempt_blocked_readers_cgp(rnp)) {

			/* Other bits still set at this level, so done. */
			raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
			return;
		}
		rnp->completedqs = rnp->gp_seq;
		mask = rnp->grpmask;
		if (rnp->parent == NULL) {

			/* No more levels.  Exit loop holding root lock. */

			break;
		}
		raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
		rnp_c = rnp;
		rnp = rnp->parent;
		raw_spin_lock_irqsave_rcu_node(rnp, flags);
		oldmask = READ_ONCE(rnp_c->qsmask);
	}

	/*
	 * Get here if we are the last CPU to pass through a quiescent
	 * state for this grace period.  Invoke rcu_report_qs_rsp()
	 * to clean up and start the next grace period if one is needed.
	 */
	rcu_report_qs_rsp(rsp, flags); /* releases rnp->lock. */
}

/*
 * Record a quiescent state for all tasks that were previously queued
 * on the specified rcu_node structure and that were blocking the current
 * RCU grace period.  The caller must hold the specified rnp->lock with
 * irqs disabled, and this lock is released upon return, but irqs remain
 * disabled.
 */
static void __maybe_unused
rcu_report_unblock_qs_rnp(struct rcu_state *rsp,
			  struct rcu_node *rnp, unsigned long flags)
	__releases(rnp->lock)
{
	unsigned long gps;
	unsigned long mask;
	struct rcu_node *rnp_p;

	raw_lockdep_assert_held_rcu_node(rnp);
	if (WARN_ON_ONCE(!IS_ENABLED(CONFIG_PREEMPT_RCU)) ||
	    WARN_ON_ONCE(rcu_preempt_blocked_readers_cgp(rnp)) ||
	    rnp->qsmask != 0) {
		raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
		return;  /* Still need more quiescent states! */
	}

	rnp->completedqs = rnp->gp_seq;
	rnp_p = rnp->parent;
	if (rnp_p == NULL) {
		/*
		 * Only one rcu_node structure in the tree, so don't
		 * try to report up to its nonexistent parent!
		 */
		rcu_report_qs_rsp(rsp, flags);
		return;
	}

	/* Report up the rest of the hierarchy, tracking current ->gp_seq. */
	gps = rnp->gp_seq;
	mask = rnp->grpmask;
	raw_spin_unlock_rcu_node(rnp);	/* irqs remain disabled. */
	raw_spin_lock_rcu_node(rnp_p);	/* irqs already disabled. */
	rcu_report_qs_rnp(mask, rsp, rnp_p, gps, flags);
}

/*
 * Record a quiescent state for the specified CPU to that CPU's rcu_data
 * structure.  This must be called from the specified CPU.
 */
static void
rcu_report_qs_rdp(int cpu, struct rcu_state *rsp, struct rcu_data *rdp)
{
	unsigned long flags;
	unsigned long mask;
	bool needwake;
	struct rcu_node *rnp;

	rnp = rdp->mynode;
	raw_spin_lock_irqsave_rcu_node(rnp, flags);
	if (rdp->cpu_no_qs.b.norm || rdp->gp_seq != rnp->gp_seq ||
	    rdp->gpwrap) {

		/*
		 * The grace period in which this quiescent state was
		 * recorded has ended, so don't report it upwards.
		 * We will instead need a new quiescent state that lies
		 * within the current grace period.
		 */
		rdp->cpu_no_qs.b.norm = true;	/* need qs for new gp. */
		rdp->rcu_qs_ctr_snap = __this_cpu_read(rcu_dynticks.rcu_qs_ctr);
		raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
		return;
	}
	mask = rdp->grpmask;
	if (rdp->cpu == smp_processor_id())
		rdp->core_needs_qs = false;
	if ((rnp->qsmask & mask) == 0) {
		raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
	} else {
		rdp->core_needs_qs = false;

		/*
		 * This GP can't end until cpu checks in, so all of our
		 * callbacks can be processed during the next GP.
		 */
		needwake = rcu_accelerate_cbs(rsp, rnp, rdp);

		rcu_disable_urgency_upon_qs(rdp);
		rcu_report_qs_rnp(mask, rnp, rnp->gp_seq, flags);
		/* ^^^ Released rnp->lock */
		if (needwake)
			rcu_gp_kthread_wake(rsp);
	}
}

/*
 * Check to see if there is a new grace period of which this CPU
 * is not yet aware, and if so, set up local rcu_data state for it.
 * Otherwise, see if this CPU has just passed through its first
 * quiescent state for this grace period, and record that fact if so.
 */
static void
rcu_check_quiescent_state(struct rcu_state *rsp, struct rcu_data *rdp)
{
	/* Check for grace-period ends and beginnings. */
	note_gp_changes(rsp, rdp);

	/*
	 * Does this CPU still need to do its part for current grace period?
	 * If no, return and let the other CPUs do their part as well.
	 */
	if (!rdp->core_needs_qs)
		return;

	/*
	 * Was there a quiescent state since the beginning of the grace
	 * period? If no, then exit and wait for the next call.
	 */
	if (rdp->cpu_no_qs.b.norm)
		return;

	/*
	 * Tell RCU we are done (but rcu_report_qs_rdp() will be the
	 * judge of that).
	 */
	rcu_report_qs_rdp(rdp->cpu, rsp, rdp);
}

/*
 * Trace the fact that this CPU is going offline.
 */
static void rcu_cleanup_dying_cpu(struct rcu_state *rsp)
{
	RCU_TRACE(bool blkd;)
	RCU_TRACE(struct rcu_data *rdp = this_cpu_ptr(rsp->rda);)
	RCU_TRACE(struct rcu_node *rnp = rdp->mynode;)

	if (!IS_ENABLED(CONFIG_HOTPLUG_CPU))
		return;

	RCU_TRACE(blkd = !!(rnp->qsmask & rdp->grpmask);)
	trace_rcu_grace_period(rcu_state.name, READ_ONCE(rnp->gp_seq),
			       blkd ? TPS("cpuofl") : TPS("cpuofl-bgp"));
}

/*
 * All CPUs for the specified rcu_node structure have gone offline,
 * and all tasks that were preempted within an RCU read-side critical
 * section while running on one of those CPUs have since exited their RCU
 * read-side critical section.  Some other CPU is reporting this fact with
 * the specified rcu_node structure's ->lock held and interrupts disabled.
 * This function therefore goes up the tree of rcu_node structures,
 * clearing the corresponding bits in the ->qsmaskinit fields.  Note that
 * the leaf rcu_node structure's ->qsmaskinit field has already been
 * updated.
 *
 * This function does check that the specified rcu_node structure has
 * all CPUs offline and no blocked tasks, so it is OK to invoke it
 * prematurely.  That said, invoking it after the fact will cost you
 * a needless lock acquisition.  So once it has done its work, don't
 * invoke it again.
 */
static void rcu_cleanup_dead_rnp(struct rcu_node *rnp_leaf)
{
	long mask;
	struct rcu_node *rnp = rnp_leaf;

	raw_lockdep_assert_held_rcu_node(rnp_leaf);
	if (!IS_ENABLED(CONFIG_HOTPLUG_CPU) ||
	    WARN_ON_ONCE(rnp_leaf->qsmaskinit) ||
	    WARN_ON_ONCE(rcu_preempt_has_tasks(rnp_leaf)))
		return;
	for (;;) {
		mask = rnp->grpmask;
		rnp = rnp->parent;
		if (!rnp)
			break;
		raw_spin_lock_rcu_node(rnp); /* irqs already disabled. */
		rnp->qsmaskinit &= ~mask;
		/* Between grace periods, so better already be zero! */
		WARN_ON_ONCE(rnp->qsmask);
		if (rnp->qsmaskinit) {
			raw_spin_unlock_rcu_node(rnp);
			/* irqs remain disabled. */
			return;
		}
		raw_spin_unlock_rcu_node(rnp); /* irqs remain disabled. */
	}
}

/*
 * The CPU has been completely removed, and some other CPU is reporting
 * this fact from process context.  Do the remainder of the cleanup.
 * There can only be one CPU hotplug operation at a time, so no need for
 * explicit locking.
 */
static void rcu_cleanup_dead_cpu(int cpu, struct rcu_state *rsp)
{
	struct rcu_data *rdp = per_cpu_ptr(rsp->rda, cpu);
	struct rcu_node *rnp = rdp->mynode;  /* Outgoing CPU's rdp & rnp. */

	if (!IS_ENABLED(CONFIG_HOTPLUG_CPU))
		return;

	/* Adjust any no-longer-needed kthreads. */
	rcu_boost_kthread_setaffinity(rnp, -1);
	/* Do any needed no-CB deferred wakeups from this CPU. */
	do_nocb_deferred_wakeup(per_cpu_ptr(&rcu_data, cpu));

	// Stop-machine done, so allow nohz_full to disable tick.
	tick_dep_clear(TICK_DEP_BIT_RCU);
	return 0;
}

/*
 * Invoke any RCU callbacks that have made it to the end of their grace
 * period.  Thottle as specified by rdp->blimit.
 */
static void rcu_do_batch(struct rcu_state *rsp, struct rcu_data *rdp)
{
	unsigned long flags;
	struct rcu_head *rhp;
	struct rcu_cblist rcl = RCU_CBLIST_INITIALIZER(rcl);
	long bl, count;

	/* If no callbacks are ready, just return. */
	if (!rcu_segcblist_ready_cbs(&rdp->cblist)) {
		trace_rcu_batch_start(rcu_state.name,
				      rcu_segcblist_n_cbs(&rdp->cblist), 0);
		trace_rcu_batch_end(rsp->name, 0,
				    !rcu_segcblist_empty(&rdp->cblist),
				    need_resched(), is_idle_task(current),
				    rcu_is_callbacks_kthread());
		return;
	}

	/*
	 * Extract the list of ready callbacks, disabling to prevent
	 * races with call_rcu() from interrupt handlers.  Leave the
	 * callback counts, as rcu_barrier() needs to be conservative.
	 */
	local_irq_save(flags);
	WARN_ON_ONCE(cpu_is_offline(smp_processor_id()));
	pending = rcu_segcblist_n_cbs(&rdp->cblist);
	bl = max(rdp->blimit, pending >> rcu_divisor);
	if (unlikely(bl > 100))
		tlimit = local_clock() + rcu_resched_ns;
	trace_rcu_batch_start(rcu_state.name,
			      rcu_segcblist_n_cbs(&rdp->cblist), bl);
	rcu_segcblist_extract_done_cbs(&rdp->cblist, &rcl);
	local_irq_restore(flags);

	/* Invoke callbacks. */
	tick_dep_set_task(current, TICK_DEP_BIT_RCU);
	rhp = rcu_cblist_dequeue(&rcl);
	for (; rhp; rhp = rcu_cblist_dequeue(&rcl)) {
		rcu_callback_t f;

		debug_rcu_head_unqueue(rhp);

		rcu_lock_acquire(&rcu_callback_map);
		trace_rcu_invoke_callback(rcu_state.name, rhp);

		f = rhp->func;
		WRITE_ONCE(rhp->func, (rcu_callback_t)0L);
		f(rhp);

		rcu_lock_release(&rcu_callback_map);

		/*
		 * Stop only if limit reached and CPU has something to do.
		 * Note: The rcl structure counts down from zero.
		 */
		if (-rcl.len >= bl &&
		    (need_resched() ||
		     (!is_idle_task(current) && !rcu_is_callbacks_kthread())))
			break;
	}

	local_irq_save(flags);
	count = -rcl.len;
	rdp->n_cbs_invoked += count;
	trace_rcu_batch_end(rcu_state.name, count, !!rcl.head, need_resched(),
			    is_idle_task(current), rcu_is_callbacks_kthread());

	/* Update counts and requeue any remaining callbacks. */
	rcu_segcblist_insert_done_cbs(&rdp->cblist, &rcl);
	smp_mb(); /* List handling before counting for rcu_barrier(). */
	rcu_segcblist_insert_count(&rdp->cblist, &rcl);

	/* Reinstate batch limit if we have worked down the excess. */
	count = rcu_segcblist_n_cbs(&rdp->cblist);
	if (rdp->blimit == LONG_MAX && count <= qlowmark)
		rdp->blimit = blimit;

	/* Reset ->qlen_last_fqs_check trigger if enough CBs have drained. */
	if (count == 0 && rdp->qlen_last_fqs_check != 0) {
		rdp->qlen_last_fqs_check = 0;
		rdp->n_force_qs_snap = rsp->n_force_qs;
	} else if (count < rdp->qlen_last_fqs_check - qhimark)
		rdp->qlen_last_fqs_check = count;

	/*
	 * The following usually indicates a double call_rcu().  To track
	 * this down, try building with CONFIG_DEBUG_OBJECTS_RCU_HEAD=y.
	 */
	WARN_ON_ONCE(rcu_segcblist_empty(&rdp->cblist) != (count == 0));

	local_irq_restore(flags);

	/* Re-invoke RCU core processing if there are callbacks remaining. */
	if (rcu_segcblist_ready_cbs(&rdp->cblist))
		invoke_rcu_core();
	tick_dep_clear_task(current, TICK_DEP_BIT_RCU);
}

/*
 * Check to see if this CPU is in a non-context-switch quiescent state
 * (user mode or idle loop for rcu, non-softirq execution for rcu_bh).
 * Also schedule RCU core processing.
 *
 * This function must be called from hardirq context.  It is normally
 * invoked from the scheduling-clock interrupt.
 */
void rcu_check_callbacks(int user)
{
	trace_rcu_utilization(TPS("Start scheduler-tick"));
	increment_cpu_stall_ticks();
	if (user || rcu_is_cpu_rrupt_from_idle()) {

		/*
		 * Get here if this CPU took its interrupt from user
		 * mode or from the idle loop, and if this is not a
		 * nested interrupt.  In this case, the CPU is in
		 * a quiescent state, so note it.
		 *
		 * No memory barrier is required here because both
		 * rcu_sched_qs() and rcu_bh_qs() reference only CPU-local
		 * variables that other CPUs neither access nor modify,
		 * at least not while the corresponding CPU is online.
		 */

		rcu_sched_qs();
		rcu_bh_qs();
		rcu_note_voluntary_context_switch(current);

	} else if (!in_softirq()) {

		/*
		 * Get here if this CPU did not take its interrupt from
		 * softirq, in other words, if it is not interrupting
		 * a rcu_bh read-side critical section.  This is an _bh
		 * critical section, so note it.
		 */

		rcu_bh_qs();
	}
	rcu_preempt_check_callbacks();
	/* The load-acquire pairs with the store-release setting to true. */
	if (smp_load_acquire(this_cpu_ptr(&rcu_dynticks.rcu_urgent_qs))) {
		/* Idle and userspace execution already are quiescent states. */
		if (!rcu_is_cpu_rrupt_from_idle() && !user) {
			set_tsk_need_resched(current);
			set_preempt_need_resched();
		}
		__this_cpu_write(rcu_dynticks.rcu_urgent_qs, false);
	}
	if (rcu_pending(user))
		invoke_rcu_core();

	trace_rcu_utilization(TPS("End scheduler-tick"));
}

/*
 * Scan the leaf rcu_node structures, processing dyntick state for any that
 * have not yet encountered a quiescent state, using the function specified.
 * Also initiate boosting for any threads blocked on the root rcu_node.
 *
 * The caller must have suppressed start of new grace periods.
 */
static void force_qs_rnp(struct rcu_state *rsp, int (*f)(struct rcu_data *rsp))
{
	int cpu;
	unsigned long flags;
	unsigned long mask;
	struct rcu_data *rdp;
	struct rcu_node *rnp;

	rcu_state.cbovld = rcu_state.cbovldnext;
	rcu_state.cbovldnext = false;
	rcu_for_each_leaf_node(rnp) {
		cond_resched_tasks_rcu_qs();
		mask = 0;
		raw_spin_lock_irqsave_rcu_node(rnp, flags);
		rcu_state.cbovldnext |= !!rnp->cbovldmask;
		if (rnp->qsmask == 0) {
			if (!IS_ENABLED(CONFIG_PREEMPT_RCU) ||
			    rcu_preempt_blocked_readers_cgp(rnp)) {
				/*
				 * No point in scanning bits because they
				 * are all zero.  But we might need to
				 * priority-boost blocked readers.
				 */
				rcu_initiate_boost(rnp, flags);
				/* rcu_initiate_boost() releases rnp->lock */
				continue;
			}
			raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
			continue;
		}
		for_each_leaf_node_cpu_mask(rnp, cpu, rnp->qsmask) {
			rdp = per_cpu_ptr(&rcu_data, cpu);
			if (f(rdp)) {
				mask |= rdp->grpmask;
				rcu_disable_urgency_upon_qs(rdp);
			}
		}
		if (mask != 0) {
			/* Idle/offline CPUs, report (releases rnp->lock). */
			rcu_report_qs_rnp(mask, rsp, rnp, rnp->gp_seq, flags);
		} else {
			/* Nothing to do here, so just drop the lock. */
			raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
		}
	}
}

/*
 * Force quiescent states on reluctant CPUs, and also detect which
 * CPUs are in dyntick-idle mode.
 */
static void force_quiescent_state(struct rcu_state *rsp)
{
	unsigned long flags;
	bool ret;
	struct rcu_node *rnp;
	struct rcu_node *rnp_old = NULL;

	/* Funnel through hierarchy to reduce memory contention. */
	rnp = __this_cpu_read(rsp->rda->mynode);
	for (; rnp != NULL; rnp = rnp->parent) {
		ret = (READ_ONCE(rcu_state.gp_flags) & RCU_GP_FLAG_FQS) ||
		       !raw_spin_trylock(&rnp->fqslock);
		if (rnp_old != NULL)
			raw_spin_unlock(&rnp_old->fqslock);
		if (ret)
			return;
		rnp_old = rnp;
	}
	/* rnp_old == rcu_get_root(rsp), rnp == NULL. */

	/* Reached the root of the rcu_node tree, acquire lock. */
	raw_spin_lock_irqsave_rcu_node(rnp_old, flags);
	raw_spin_unlock(&rnp_old->fqslock);
	if (READ_ONCE(rsp->gp_flags) & RCU_GP_FLAG_FQS) {
		raw_spin_unlock_irqrestore_rcu_node(rnp_old, flags);
		return;  /* Someone beat us to it. */
	}
	WRITE_ONCE(rsp->gp_flags, READ_ONCE(rsp->gp_flags) | RCU_GP_FLAG_FQS);
	raw_spin_unlock_irqrestore_rcu_node(rnp_old, flags);
	rcu_gp_kthread_wake(rsp);
}

/*
 * This function checks for grace-period requests that fail to motivate
 * RCU to come out of its idle mode.
 */
static void
rcu_check_gp_start_stall(struct rcu_state *rsp, struct rcu_node *rnp,
			 struct rcu_data *rdp)
{
	const unsigned long gpssdelay = rcu_jiffies_till_stall_check() * HZ;
	unsigned long flags;
	unsigned long j;
	struct rcu_node *rnp_root = rcu_get_root(rsp);
	static atomic_t warned = ATOMIC_INIT(0);

	if (!IS_ENABLED(CONFIG_PROVE_RCU) || rcu_gp_in_progress(rsp) ||
	    ULONG_CMP_GE(rnp_root->gp_seq, rnp_root->gp_seq_needed))
		return;
	j = jiffies; /* Expensive access, and in common case don't get here. */
	if (time_before(j, READ_ONCE(rsp->gp_req_activity) + gpssdelay) ||
	    time_before(j, READ_ONCE(rsp->gp_activity) + gpssdelay) ||
	    atomic_read(&warned))
		return;

	raw_spin_lock_irqsave_rcu_node(rnp, flags);
	j = jiffies;
	if (rcu_gp_in_progress(rsp) ||
	    ULONG_CMP_GE(rnp_root->gp_seq, rnp_root->gp_seq_needed) ||
	    time_before(j, READ_ONCE(rsp->gp_req_activity) + gpssdelay) ||
	    time_before(j, READ_ONCE(rsp->gp_activity) + gpssdelay) ||
	    atomic_read(&warned)) {
		raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
		return;
	}
	/* Hold onto the leaf lock to make others see warned==1. */

	if (rnp_root != rnp)
		raw_spin_lock_rcu_node(rnp_root); /* irqs already disabled. */
	j = jiffies;
	if (rcu_gp_in_progress(rsp) ||
	    ULONG_CMP_GE(rnp_root->gp_seq, rnp_root->gp_seq_needed) ||
	    time_before(j, rsp->gp_req_activity + gpssdelay) ||
	    time_before(j, rsp->gp_activity + gpssdelay) ||
	    atomic_xchg(&warned, 1)) {
		raw_spin_unlock_rcu_node(rnp_root); /* irqs remain disabled. */
		raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
		return;
	}
	pr_alert("%s: g%ld->%ld gar:%lu ga:%lu f%#x gs:%d %s->state:%#lx\n",
		 __func__, (long)READ_ONCE(rsp->gp_seq),
		 (long)READ_ONCE(rnp_root->gp_seq_needed),
		 j - rsp->gp_req_activity, j - rsp->gp_activity,
		 rsp->gp_flags, rsp->gp_state, rsp->name,
		 rsp->gp_kthread ? rsp->gp_kthread->state : 0x1ffffL);
	WARN_ON(1);
	if (rnp_root != rnp)
		raw_spin_unlock_rcu_node(rnp_root);
	raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
}

/*
 * This does the RCU core processing work for the specified rcu_state
 * and rcu_data structures.  This may be called only from the CPU to
 * whom the rdp belongs.
 */
static void
__rcu_process_callbacks(struct rcu_state *rsp)
{
	unsigned long flags;
	struct rcu_data *rdp = raw_cpu_ptr(rsp->rda);
	struct rcu_node *rnp = rdp->mynode;

	WARN_ON_ONCE(!rdp->beenonline);

	/* Update RCU state based on any recent quiescent states. */
	rcu_check_quiescent_state(rsp, rdp);

	/* No grace period and unregistered callbacks? */
	if (!rcu_gp_in_progress(rsp) &&
	    rcu_segcblist_is_enabled(&rdp->cblist)) {
		local_irq_save(flags);
		if (!rcu_segcblist_restempty(&rdp->cblist, RCU_NEXT_READY_TAIL))
			rcu_accelerate_cbs_unlocked(rsp, rnp, rdp);
		local_irq_restore(flags);
	}

	rcu_check_gp_start_stall(rsp, rnp, rdp);

	/* If there are callbacks ready, invoke them. */
	if (rcu_segcblist_ready_cbs(&rdp->cblist))
		invoke_rcu_callbacks(rsp, rdp);

	/* Do any needed deferred wakeups of rcuo kthreads. */
	do_nocb_deferred_wakeup(rdp);
}

/*
 * Do RCU core processing for the current CPU.
 */
static __latent_entropy void rcu_process_callbacks(struct softirq_action *unused)
{
	struct rcu_state *rsp;

	if (cpu_is_offline(smp_processor_id()))
		return;
	trace_rcu_utilization(TPS("Start RCU core"));
	for_each_rcu_flavor(rsp)
		__rcu_process_callbacks(rsp);
	trace_rcu_utilization(TPS("End RCU core"));
}

/*
 * Schedule RCU callback invocation.  If the specified type of RCU
 * does not support RCU priority boosting, just do a direct call,
 * otherwise wake up the per-CPU kernel kthread.  Note that because we
 * are running on the current CPU with softirqs disabled, the
 * rcu_cpu_kthread_task cannot disappear out from under us.
 */
static void invoke_rcu_callbacks(struct rcu_state *rsp, struct rcu_data *rdp)
{
	if (unlikely(!READ_ONCE(rcu_scheduler_fully_active)))
		return;
	if (likely(!rsp->boost)) {
		rcu_do_batch(rsp, rdp);
		return;
	}
	invoke_rcu_callbacks_kthread();
}

static void invoke_rcu_core(void)
{
	if (cpu_online(smp_processor_id()))
		raise_softirq(RCU_SOFTIRQ);
}

static void rcu_cpu_kthread_park(unsigned int cpu)
{
	per_cpu(rcu_data.rcu_cpu_kthread_status, cpu) = RCU_KTHREAD_OFFCPU;
}

static int rcu_cpu_kthread_should_run(unsigned int cpu)
{
	return __this_cpu_read(rcu_data.rcu_cpu_has_work);
}

/*
 * Per-CPU kernel thread that invokes RCU callbacks.  This replaces
 * the RCU softirq used in configurations of RCU that do not support RCU
 * priority boosting.
 */
static void rcu_cpu_kthread(unsigned int cpu)
{
	unsigned int *statusp = this_cpu_ptr(&rcu_data.rcu_cpu_kthread_status);
	char work, *workp = this_cpu_ptr(&rcu_data.rcu_cpu_has_work);
	int spincnt;

	trace_rcu_utilization(TPS("Start CPU kthread@rcu_run"));
	for (spincnt = 0; spincnt < 10; spincnt++) {
		local_bh_disable();
		*statusp = RCU_KTHREAD_RUNNING;
		local_irq_disable();
		work = *workp;
		*workp = 0;
		local_irq_enable();
		if (work)
			rcu_core();
		local_bh_enable();
		if (*workp == 0) {
			trace_rcu_utilization(TPS("End CPU kthread@rcu_wait"));
			*statusp = RCU_KTHREAD_WAITING;
			return;
		}
	}
	*statusp = RCU_KTHREAD_YIELDING;
	trace_rcu_utilization(TPS("Start CPU kthread@rcu_yield"));
	schedule_timeout_idle(2);
	trace_rcu_utilization(TPS("End CPU kthread@rcu_yield"));
	*statusp = RCU_KTHREAD_WAITING;
}

static struct smp_hotplug_thread rcu_cpu_thread_spec = {
	.store			= &rcu_data.rcu_cpu_kthread_task,
	.thread_should_run	= rcu_cpu_kthread_should_run,
	.thread_fn		= rcu_cpu_kthread,
	.thread_comm		= "rcuc/%u",
	.setup			= rcu_cpu_kthread_setup,
	.park			= rcu_cpu_kthread_park,
};

/*
 * Spawn per-CPU RCU core processing kthreads.
 */
static int __init rcu_spawn_core_kthreads(void)
{
	int cpu;

	for_each_possible_cpu(cpu)
		per_cpu(rcu_data.rcu_cpu_has_work, cpu) = 0;
	if (!IS_ENABLED(CONFIG_RCU_BOOST) && use_softirq)
		return 0;
	WARN_ONCE(smpboot_register_percpu_thread(&rcu_cpu_thread_spec),
		  "%s: Could not start rcuc kthread, OOM is now expected behavior\n", __func__);
	return 0;
}
early_initcall(rcu_spawn_core_kthreads);

/*
 * Handle any core-RCU processing required by a call_rcu() invocation.
 */
static void __call_rcu_core(struct rcu_state *rsp, struct rcu_data *rdp,
			    struct rcu_head *head, unsigned long flags)
{
	/*
	 * If called from an extended quiescent state, invoke the RCU
	 * core in order to force a re-evaluation of RCU's idleness.
	 */
	if (!rcu_is_watching())
		invoke_rcu_core();

	/* If interrupts were disabled or CPU offline, don't invoke RCU core. */
	if (irqs_disabled_flags(flags) || cpu_is_offline(smp_processor_id()))
		return;

	/*
	 * Force the grace period if too many callbacks or too long waiting.
	 * Enforce hysteresis, and don't invoke force_quiescent_state()
	 * if some other CPU has recently done so.  Also, don't bother
	 * invoking force_quiescent_state() if the newly enqueued callback
	 * is the only one waiting for a grace period to complete.
	 */
	if (unlikely(rcu_segcblist_n_cbs(&rdp->cblist) >
		     rdp->qlen_last_fqs_check + qhimark)) {

		/* Are we ignoring a completed grace period? */
		note_gp_changes(rsp, rdp);

		/* Start a new grace period if one not already started. */
		if (!rcu_gp_in_progress(rsp)) {
			rcu_accelerate_cbs_unlocked(rsp, rdp->mynode, rdp);
		} else {
			/* Give the grace period a kick. */
			rdp->blimit = LONG_MAX;
			if (rsp->n_force_qs == rdp->n_force_qs_snap &&
			    rcu_segcblist_first_pend_cb(&rdp->cblist) != head)
				force_quiescent_state(rsp);
			rdp->n_force_qs_snap = rsp->n_force_qs;
			rdp->qlen_last_fqs_check = rcu_segcblist_n_cbs(&rdp->cblist);
		}
	}
}

/*
 * RCU callback function to leak a callback.
 */
static void rcu_leak_callback(struct rcu_head *rhp)
{
}

/*
 * Check and if necessary update the leaf rcu_node structure's
 * ->cbovldmask bit corresponding to the current CPU based on that CPU's
 * number of queued RCU callbacks.  The caller must hold the leaf rcu_node
 * structure's ->lock.
 */
static void check_cb_ovld_locked(struct rcu_data *rdp, struct rcu_node *rnp)
{
	raw_lockdep_assert_held_rcu_node(rnp);
	if (qovld_calc <= 0)
		return; // Early boot and wildcard value set.
	if (rcu_segcblist_n_cbs(&rdp->cblist) >= qovld_calc)
		WRITE_ONCE(rnp->cbovldmask, rnp->cbovldmask | rdp->grpmask);
	else
		WRITE_ONCE(rnp->cbovldmask, rnp->cbovldmask & ~rdp->grpmask);
}

/*
 * Check and if necessary update the leaf rcu_node structure's
 * ->cbovldmask bit corresponding to the current CPU based on that CPU's
 * number of queued RCU callbacks.  No locks need be held, but the
 * caller must have disabled interrupts.
 *
 * Note that this function ignores the possibility that there are a lot
 * of callbacks all of which have already seen the end of their respective
 * grace periods.  This omission is due to the need for no-CBs CPUs to
 * be holding ->nocb_lock to do this check, which is too heavy for a
 * common-case operation.
 */
static void check_cb_ovld(struct rcu_data *rdp)
{
	struct rcu_node *const rnp = rdp->mynode;

	if (qovld_calc <= 0 ||
	    ((rcu_segcblist_n_cbs(&rdp->cblist) >= qovld_calc) ==
	     !!(READ_ONCE(rnp->cbovldmask) & rdp->grpmask)))
		return; // Early boot wildcard value or already set correctly.
	raw_spin_lock_rcu_node(rnp);
	check_cb_ovld_locked(rdp, rnp);
	raw_spin_unlock_rcu_node(rnp);
}

/* Helper function for call_rcu() and friends.  */
static void
__call_rcu(struct rcu_head *head, rcu_callback_t func)
{
	unsigned long flags;
	struct rcu_data *rdp;

	/* Misaligned rcu_head! */
	WARN_ON_ONCE((unsigned long)head & (sizeof(void *) - 1));

	if (debug_rcu_head_queue(head)) {
		/*
		 * Probable double call_rcu(), so leak the callback.
		 * Use rcu:rcu_callback trace event to find the previous
		 * time callback was passed to __call_rcu().
		 */
		WARN_ONCE(1, "__call_rcu(): Double-freed CB %p->%pF()!!!\n",
			  head, head->func);
		WRITE_ONCE(head->func, rcu_leak_callback);
		return;
	}
	head->func = func;
	head->next = NULL;
	local_irq_save(flags);
	rdp = this_cpu_ptr(rsp->rda);

	/* Add the callback to our list. */
	if (unlikely(!rcu_segcblist_is_enabled(&rdp->cblist)) || cpu != -1) {
		int offline;

		if (cpu != -1)
			rdp = per_cpu_ptr(rsp->rda, cpu);
		if (likely(rdp->mynode)) {
			/* Post-boot, so this should be for a no-CBs CPU. */
			offline = !__call_rcu_nocb(rdp, head, lazy, flags);
			WARN_ON_ONCE(offline);
			/* Offline CPU, _call_rcu() illegal, leak callback.  */
			local_irq_restore(flags);
			return;
		}
		/*
		 * Very early boot, before rcu_init().  Initialize if needed
		 * and then drop through to queue the callback.
		 */
		BUG_ON(cpu != -1);
		WARN_ON_ONCE(!rcu_is_watching());
		if (rcu_segcblist_empty(&rdp->cblist))
			rcu_segcblist_init(&rdp->cblist);
	}

	check_cb_ovld(rdp);
	if (rcu_nocb_try_bypass(rdp, head, &was_alldone, flags))
		return; // Enqueued onto ->nocb_bypass, so just leave.
	// If no-CBs CPU gets here, rcu_nocb_try_bypass() acquired ->nocb_lock.
	rcu_segcblist_enqueue(&rdp->cblist, head);
	if (__is_kfree_rcu_offset((unsigned long)func))
		trace_rcu_kfree_callback(rcu_state.name, head,
					 (unsigned long)func,
					 rcu_segcblist_n_cbs(&rdp->cblist));
	else
		trace_rcu_callback(rcu_state.name, head,
				   rcu_segcblist_n_cbs(&rdp->cblist));

	/* Go handle any RCU core processing required. */
	__call_rcu_core(rsp, rdp, head, flags);
	local_irq_restore(flags);
}

/**
 * call_rcu_sched() - Queue an RCU for invocation after sched grace period.
 * @head: structure to be used for queueing the RCU updates.
 * @func: actual callback function to be invoked after the grace period
 *
 * The callback function will be invoked some time after a full grace
 * period elapses, in other words after all currently executing RCU
 * read-side critical sections have completed. call_rcu_sched() assumes
 * that the read-side critical sections end on enabling of preemption
 * or on voluntary preemption.
 * RCU read-side critical sections are delimited by:
 *
 * - rcu_read_lock_sched() and rcu_read_unlock_sched(), OR
 * - anything that disables preemption.
 *
 *  These may be nested.
 *
 * See the description of call_rcu() for more detailed information on
 * memory ordering guarantees.
 */
void call_rcu_sched(struct rcu_head *head, rcu_callback_t func)
{
	__call_rcu(head, func);
}
EXPORT_SYMBOL_GPL(call_rcu_sched);

/**
 * call_rcu_bh() - Queue an RCU for invocation after a quicker grace period.
 * @head: structure to be used for queueing the RCU updates.
 * @func: actual callback function to be invoked after the grace period
 *
 * The callback function will be invoked some time after a full grace
 * period elapses, in other words after all currently executing RCU
 * read-side critical sections have completed. call_rcu_bh() assumes
 * that the read-side critical sections end on completion of a softirq
 * handler. This means that read-side critical sections in process
 * context must not be interrupted by softirqs. This interface is to be
 * used when most of the read-side critical sections are in softirq context.
 * RCU read-side critical sections are delimited by:
 *
 * - rcu_read_lock() and  rcu_read_unlock(), if in interrupt context, OR
 * - rcu_read_lock_bh() and rcu_read_unlock_bh(), if in process context.
 *
 * These may be nested.
 *
 * See the description of call_rcu() for more detailed information on
 * memory ordering guarantees.
 */
void call_rcu_bh(struct rcu_head *head, rcu_callback_t func)
{
	__call_rcu(head, func, &rcu_bh_state, -1, 0);
}
EXPORT_SYMBOL_GPL(call_rcu_bh);


/* Maximum number of jiffies to wait before draining a batch. */
#define KFREE_DRAIN_JIFFIES (HZ / 50)
#define KFREE_N_BATCHES 2

/**
 * struct kfree_rcu_bulk_data - single block to store kfree_rcu() pointers
 * @nr_records: Number of active pointers in the array
 * @records: Array of the kfree_rcu() pointers
 * @next: Next bulk object in the block chain
 */
struct kfree_rcu_bulk_data {
	unsigned long nr_records;
	struct kfree_rcu_bulk_data *next;
	void *records[];
};

/*
 * This macro defines how many entries the "records" array
 * will contain. It is based on the fact that the size of
 * kfree_rcu_bulk_data structure becomes exactly one page.
 */
#define KFREE_BULK_MAX_ENTR \
	((PAGE_SIZE - sizeof(struct kfree_rcu_bulk_data)) / sizeof(void *))

/**
 * struct kfree_rcu_cpu_work - single batch of kfree_rcu() requests
 * @rcu_work: Let queue_rcu_work() invoke workqueue handler after grace period
 * @head_free: List of kfree_rcu() objects waiting for a grace period
 * @bhead_free: Bulk-List of kfree_rcu() objects waiting for a grace period
 * @krcp: Pointer to @kfree_rcu_cpu structure
 */

struct kfree_rcu_cpu_work {
	struct rcu_work rcu_work;
	struct rcu_head *head_free;
	struct kfree_rcu_bulk_data *bhead_free;
	struct kfree_rcu_cpu *krcp;
};

/**
 * struct kfree_rcu_cpu - batch up kfree_rcu() requests for RCU grace period
 * @head: List of kfree_rcu() objects not yet waiting for a grace period
 * @bhead: Bulk-List of kfree_rcu() objects not yet waiting for a grace period
 * @krw_arr: Array of batches of kfree_rcu() objects waiting for a grace period
 * @lock: Synchronize access to this structure
 * @monitor_work: Promote @head to @head_free after KFREE_DRAIN_JIFFIES
 * @monitor_todo: Tracks whether a @monitor_work delayed work is pending
 * @initialized: The @rcu_work fields have been initialized
 * @count: Number of objects for which GP not started
 *
 * This is a per-CPU structure.  The reason that it is not included in
 * the rcu_data structure is to permit this code to be extracted from
 * the RCU files.  Such extraction could allow further optimization of
 * the interactions with the slab allocators.
 */
struct kfree_rcu_cpu {
	struct rcu_head *head;
	struct kfree_rcu_bulk_data *bhead;
	struct kfree_rcu_cpu_work krw_arr[KFREE_N_BATCHES];
	raw_spinlock_t lock;
	struct delayed_work monitor_work;
	bool monitor_todo;
	bool initialized;
	int count;

	/*
	 * A simple cache list that contains objects for
	 * reuse purpose. In order to save some per-cpu
	 * space the list is singular. Even though it is
	 * lockless an access has to be protected by the
	 * per-cpu lock.
	 */
	struct llist_head bkvcache;
	int nr_bkv_objs;
};

static DEFINE_PER_CPU(struct kfree_rcu_cpu, krc) = {
	.lock = __RAW_SPIN_LOCK_UNLOCKED(krc.lock),
};

static __always_inline void
debug_rcu_bhead_unqueue(struct kfree_rcu_bulk_data *bhead)
{
#ifdef CONFIG_DEBUG_OBJECTS_RCU_HEAD
	int i;

	for (i = 0; i < bhead->nr_records; i++)
		debug_rcu_head_unqueue((struct rcu_head *)(bhead->records[i]));
#endif
}

static inline struct kfree_rcu_cpu *
krc_this_cpu_lock(unsigned long *flags)
{
	struct kfree_rcu_cpu *krcp;

	local_irq_save(*flags);	// For safely calling this_cpu_ptr().
	krcp = this_cpu_ptr(&krc);
	raw_spin_lock(&krcp->lock);

	return krcp;
}

static inline void
krc_this_cpu_unlock(struct kfree_rcu_cpu *krcp, unsigned long flags)
{
	raw_spin_unlock(&krcp->lock);
	local_irq_restore(flags);
}

static inline struct kfree_rcu_bulk_data *
get_cached_bnode(struct kfree_rcu_cpu *krcp)
{
	if (!krcp->nr_bkv_objs)
		return NULL;

	krcp->nr_bkv_objs--;
	return (struct kfree_rcu_bulk_data *)
		llist_del_first(&krcp->bkvcache);
}

static inline bool
put_cached_bnode(struct kfree_rcu_cpu *krcp,
	struct kfree_rcu_bulk_data *bnode)
{
	// Check the limit.
	if (krcp->nr_bkv_objs >= rcu_min_cached_objs)
		return false;

	llist_add((struct llist_node *) bnode, &krcp->bkvcache);
	krcp->nr_bkv_objs++;
	return true;

}

/*
 * This function is invoked in workqueue context after a grace period.
 * It frees all the objects queued on ->bhead_free or ->head_free.
 */
static void kfree_rcu_work(struct work_struct *work)
{
	unsigned long flags;
	struct rcu_head *head, *next;
	struct kfree_rcu_bulk_data *bhead, *bnext;
	struct kfree_rcu_cpu *krcp;
	struct kfree_rcu_cpu_work *krwp;

	krwp = container_of(to_rcu_work(work),
			    struct kfree_rcu_cpu_work, rcu_work);
	krcp = krwp->krcp;
	raw_spin_lock_irqsave(&krcp->lock, flags);
	head = krwp->head_free;
	krwp->head_free = NULL;
	bhead = krwp->bhead_free;
	krwp->bhead_free = NULL;
	raw_spin_unlock_irqrestore(&krcp->lock, flags);

	/* "bhead" is now private, so traverse locklessly. */
	for (; bhead; bhead = bnext) {
		bnext = bhead->next;

		debug_rcu_bhead_unqueue(bhead);

		rcu_lock_acquire(&rcu_callback_map);
		trace_rcu_invoke_kfree_bulk_callback(rcu_state.name,
			bhead->nr_records, bhead->records);

		kfree_bulk(bhead->nr_records, bhead->records);
		rcu_lock_release(&rcu_callback_map);

		krcp = krc_this_cpu_lock(&flags);
		if (put_cached_bnode(krcp, bhead))
			bhead = NULL;
		krc_this_cpu_unlock(krcp, flags);

		if (bhead)
			free_page((unsigned long) bhead);

		cond_resched_tasks_rcu_qs();
	}

	/*
	 * Emergency case only. It can happen under low memory
	 * condition when an allocation gets failed, so the "bulk"
	 * path can not be temporary maintained.
	 */
	for (; head; head = next) {
		unsigned long offset = (unsigned long)head->func;
		void *ptr = (void *)head - offset;

		next = head->next;
		debug_rcu_head_unqueue((struct rcu_head *)ptr);
		rcu_lock_acquire(&rcu_callback_map);
		trace_rcu_invoke_kfree_callback(rcu_state.name, head, offset);

		if (!WARN_ON_ONCE(!__is_kfree_rcu_offset(offset)))
			kfree(ptr);

		rcu_lock_release(&rcu_callback_map);
		cond_resched_tasks_rcu_qs();
	}
}

/*
 * Schedule the kfree batch RCU work to run in workqueue context after a GP.
 *
 * This function is invoked by kfree_rcu_monitor() when the KFREE_DRAIN_JIFFIES
 * timeout has been reached.
 */
static inline bool queue_kfree_rcu_work(struct kfree_rcu_cpu *krcp)
{
	struct kfree_rcu_cpu_work *krwp;
	bool repeat = false;
	int i;

	lockdep_assert_held(&krcp->lock);

	for (i = 0; i < KFREE_N_BATCHES; i++) {
		krwp = &(krcp->krw_arr[i]);

		/*
		 * Try to detach bhead or head and attach it over any
		 * available corresponding free channel. It can be that
		 * a previous RCU batch is in progress, it means that
		 * immediately to queue another one is not possible so
		 * return false to tell caller to retry.
		 */
		if ((krcp->bhead && !krwp->bhead_free) ||
				(krcp->head && !krwp->head_free)) {
			/* Channel 1. */
			if (!krwp->bhead_free) {
				krwp->bhead_free = krcp->bhead;
				krcp->bhead = NULL;
			}

			/* Channel 2. */
			if (!krwp->head_free) {
				krwp->head_free = krcp->head;
				krcp->head = NULL;
			}

			WRITE_ONCE(krcp->count, 0);

			/*
			 * One work is per one batch, so there are two "free channels",
			 * "bhead_free" and "head_free" the batch can handle. It can be
			 * that the work is in the pending state when two channels have
			 * been detached following each other, one by one.
			 */
			queue_rcu_work(system_wq, &krwp->rcu_work);
		}

		/* Repeat if any "free" corresponding channel is still busy. */
		if (krcp->bhead || krcp->head)
			repeat = true;
	}

	return !repeat;
}

static inline void kfree_rcu_drain_unlock(struct kfree_rcu_cpu *krcp,
					  unsigned long flags)
{
	// Attempt to start a new batch.
	krcp->monitor_todo = false;
	if (queue_kfree_rcu_work(krcp)) {
		// Success! Our job is done here.
		raw_spin_unlock_irqrestore(&krcp->lock, flags);
		return;
	}

	// Previous RCU batch still in progress, try again later.
	krcp->monitor_todo = true;
	schedule_delayed_work(&krcp->monitor_work, KFREE_DRAIN_JIFFIES);
	raw_spin_unlock_irqrestore(&krcp->lock, flags);
}

/*
 * This function is invoked after the KFREE_DRAIN_JIFFIES timeout.
 * It invokes kfree_rcu_drain_unlock() to attempt to start another batch.
 */
static void kfree_rcu_monitor(struct work_struct *work)
{
	unsigned long flags;
	struct kfree_rcu_cpu *krcp = container_of(work, struct kfree_rcu_cpu,
						 monitor_work.work);

	raw_spin_lock_irqsave(&krcp->lock, flags);
	if (krcp->monitor_todo)
		kfree_rcu_drain_unlock(krcp, flags);
	else
		raw_spin_unlock_irqrestore(&krcp->lock, flags);
}

static inline bool
kfree_call_rcu_add_ptr_to_bulk(struct kfree_rcu_cpu *krcp,
	struct rcu_head *head, rcu_callback_t func)
{
	struct kfree_rcu_bulk_data *bnode;

	if (unlikely(!krcp->initialized))
		return false;

	lockdep_assert_held(&krcp->lock);

	/* Check if a new block is required. */
	if (!krcp->bhead ||
			krcp->bhead->nr_records == KFREE_BULK_MAX_ENTR) {
		bnode = get_cached_bnode(krcp);
		if (!bnode) {
			WARN_ON_ONCE(sizeof(struct kfree_rcu_bulk_data) > PAGE_SIZE);

			/*
			 * To keep this path working on raw non-preemptible
			 * sections, prevent the optional entry into the
			 * allocator as it uses sleeping locks. In fact, even
			 * if the caller of kfree_rcu() is preemptible, this
			 * path still is not, as krcp->lock is a raw spinlock.
			 * With additional page pre-allocation in the works,
			 * hitting this return is going to be much less likely.
			 */
			if (IS_ENABLED(CONFIG_PREEMPT_RT))
				return false;

			bnode = (struct kfree_rcu_bulk_data *)
				__get_free_page(GFP_NOWAIT | __GFP_NOWARN);
		}

		/* Switch to emergency path. */
		if (unlikely(!bnode))
			return false;

		/* Initialize the new block. */
		bnode->nr_records = 0;
		bnode->next = krcp->bhead;

		/* Attach it to the head. */
		krcp->bhead = bnode;
	}

	/* Finally insert. */
	krcp->bhead->records[krcp->bhead->nr_records++] =
		(void *) head - (unsigned long) func;

	return true;
}

/*
 * Queue a request for lazy invocation of kfree_bulk()/kfree() after a grace
 * period. Please note there are two paths are maintained, one is the main one
 * that uses kfree_bulk() interface and second one is emergency one, that is
 * used only when the main path can not be maintained temporary, due to memory
 * pressure.
 *
 * Each kfree_call_rcu() request is added to a batch. The batch will be drained
 * every KFREE_DRAIN_JIFFIES number of jiffies. All the objects in the batch will
 * be free'd in workqueue context. This allows us to: batch requests together to
 * reduce the number of grace periods during heavy kfree_rcu() load.
 */
void kfree_call_rcu(struct rcu_head *head, rcu_callback_t func)
{
	unsigned long flags;
	struct kfree_rcu_cpu *krcp;
	void *ptr;

	krcp = krc_this_cpu_lock(&flags);
	ptr = (void *)head - (unsigned long)func;

	// Queue the object but don't yet schedule the batch.
	if (debug_rcu_head_queue(ptr)) {
		// Probable double kfree_rcu(), just leak.
		WARN_ONCE(1, "%s(): Double-freed call. rcu_head %p\n",
			  __func__, head);
		goto unlock_return;
	}

	/*
	 * Under high memory pressure GFP_NOWAIT can fail,
	 * in that case the emergency path is maintained.
	 */
	if (unlikely(!kfree_call_rcu_add_ptr_to_bulk(krcp, head, func))) {
		head->func = func;
		head->next = krcp->head;
		krcp->head = head;
	}

	WRITE_ONCE(krcp->count, krcp->count + 1);

	// Set timer to drain after KFREE_DRAIN_JIFFIES.
	if (rcu_scheduler_active == RCU_SCHEDULER_RUNNING &&
	    !krcp->monitor_todo) {
		krcp->monitor_todo = true;
		schedule_delayed_work(&krcp->monitor_work, KFREE_DRAIN_JIFFIES);
	}

unlock_return:
	krc_this_cpu_unlock(krcp, flags);
}
EXPORT_SYMBOL_GPL(kfree_call_rcu);

static unsigned long
kfree_rcu_shrink_count(struct shrinker *shrink, struct shrink_control *sc)
{
	int cpu;
	unsigned long count = 0;

	/* Snapshot count of all CPUs */
	for_each_online_cpu(cpu) {
		struct kfree_rcu_cpu *krcp = per_cpu_ptr(&krc, cpu);

		count += READ_ONCE(krcp->count);
	}

	return count;
}

static unsigned long
kfree_rcu_shrink_scan(struct shrinker *shrink, struct shrink_control *sc)
{
	int cpu, freed = 0;
	unsigned long flags;

	for_each_online_cpu(cpu) {
		int count;
		struct kfree_rcu_cpu *krcp = per_cpu_ptr(&krc, cpu);

		count = krcp->count;
		raw_spin_lock_irqsave(&krcp->lock, flags);
		if (krcp->monitor_todo)
			kfree_rcu_drain_unlock(krcp, flags);
		else
			raw_spin_unlock_irqrestore(&krcp->lock, flags);

		sc->nr_to_scan -= count;
		freed += count;

		if (sc->nr_to_scan <= 0)
			break;
	}

	return freed == 0 ? SHRINK_STOP : freed;
}

static struct shrinker kfree_rcu_shrinker = {
	.count_objects = kfree_rcu_shrink_count,
	.scan_objects = kfree_rcu_shrink_scan,
	.batch = 0,
	.seeks = DEFAULT_SEEKS,
};

void __init kfree_rcu_scheduler_running(void)
{
	int cpu;
	unsigned long flags;

	for_each_online_cpu(cpu) {
		struct kfree_rcu_cpu *krcp = per_cpu_ptr(&krc, cpu);

		raw_spin_lock_irqsave(&krcp->lock, flags);
		if (!krcp->head || krcp->monitor_todo) {
			raw_spin_unlock_irqrestore(&krcp->lock, flags);
			continue;
		}
		krcp->monitor_todo = true;
		schedule_delayed_work_on(cpu, &krcp->monitor_work,
					 KFREE_DRAIN_JIFFIES);
		raw_spin_unlock_irqrestore(&krcp->lock, flags);
	}
}

/*
 * Because a context switch is a grace period for RCU-sched and RCU-bh,
 * any blocking grace-period wait automatically implies a grace period
 * if there is only one CPU online at any point time during execution
 * of either synchronize_sched() or synchronize_rcu_bh().  It is OK to
 * occasionally incorrectly indicate that there are multiple CPUs online
 * when there was in fact only one the whole time, as this just adds
 * some overhead: RCU still operates correctly.
 */
static int rcu_blocking_is_gp(void)
{
	int ret;

	might_sleep();  /* Check for RCU read-side critical section. */
	preempt_disable();
	ret = num_online_cpus() <= 1;
	preempt_enable();
	return ret;
}

/**
 * synchronize_sched - wait until an rcu-sched grace period has elapsed.
 *
 * Control will return to the caller some time after a full rcu-sched
 * grace period has elapsed, in other words after all currently executing
 * rcu-sched read-side critical sections have completed.   These read-side
 * critical sections are delimited by rcu_read_lock_sched() and
 * rcu_read_unlock_sched(), and may be nested.  Note that preempt_disable(),
 * local_irq_disable(), and so on may be used in place of
 * rcu_read_lock_sched().
 *
 * This means that all preempt_disable code sequences, including NMI and
 * non-threaded hardware-interrupt handlers, in progress on entry will
 * have completed before this primitive returns.  However, this does not
 * guarantee that softirq handlers will have completed, since in some
 * kernels, these handlers can run in process context, and can block.
 *
 * Note that this guarantee implies further memory-ordering guarantees.
 * On systems with more than one CPU, when synchronize_sched() returns,
 * each CPU is guaranteed to have executed a full memory barrier since the
 * end of its last RCU-sched read-side critical section whose beginning
 * preceded the call to synchronize_sched().  In addition, each CPU having
 * an RCU read-side critical section that extends beyond the return from
 * synchronize_sched() is guaranteed to have executed a full memory barrier
 * after the beginning of synchronize_sched() and before the beginning of
 * that RCU read-side critical section.  Note that these guarantees include
 * CPUs that are offline, idle, or executing in user mode, as well as CPUs
 * that are executing in the kernel.
 *
 * Furthermore, if CPU A invoked synchronize_sched(), which returned
 * to its caller on CPU B, then both CPU A and CPU B are guaranteed
 * to have executed a full memory barrier during the execution of
 * synchronize_sched() -- even if CPU A and CPU B are the same CPU (but
 * again only if the system has more than one CPU).
 */
void synchronize_sched(void)
{
	RCU_LOCKDEP_WARN(lock_is_held(&rcu_bh_lock_map) ||
			 lock_is_held(&rcu_lock_map) ||
			 lock_is_held(&rcu_sched_lock_map),
			 "Illegal synchronize_sched() in RCU-sched read-side critical section");
	if (rcu_blocking_is_gp())
		return;
	if (rcu_gp_is_expedited())
		synchronize_sched_expedited();
	else
		wait_rcu_gp(call_rcu_sched);
}
EXPORT_SYMBOL_GPL(synchronize_sched);

/**
 * synchronize_rcu_bh - wait until an rcu_bh grace period has elapsed.
 *
 * Control will return to the caller some time after a full rcu_bh grace
 * period has elapsed, in other words after all currently executing rcu_bh
 * read-side critical sections have completed.  RCU read-side critical
 * sections are delimited by rcu_read_lock_bh() and rcu_read_unlock_bh(),
 * and may be nested.
 *
 * See the description of synchronize_sched() for more detailed information
 * on memory ordering guarantees.
 */
void synchronize_rcu_bh(void)
{
	RCU_LOCKDEP_WARN(lock_is_held(&rcu_bh_lock_map) ||
			 lock_is_held(&rcu_lock_map) ||
			 lock_is_held(&rcu_sched_lock_map),
			 "Illegal synchronize_rcu_bh() in RCU-bh read-side critical section");
	if (rcu_blocking_is_gp())
		return;
	if (rcu_gp_is_expedited())
		synchronize_rcu_bh_expedited();
	else
		wait_rcu_gp(call_rcu_bh);
}
EXPORT_SYMBOL_GPL(synchronize_rcu_bh);

/**
 * get_state_synchronize_rcu - Snapshot current RCU state
 *
 * Returns a cookie that is used by a later call to cond_synchronize_rcu()
 * to determine whether or not a full grace period has elapsed in the
 * meantime.
 */
unsigned long get_state_synchronize_rcu(void)
{
	/*
	 * Any prior manipulation of RCU-protected data must happen
	 * before the load from ->gp_seq.
	 */
	smp_mb();  /* ^^^ */
	return rcu_seq_snap(&rcu_state_p->gp_seq);
}
EXPORT_SYMBOL_GPL(get_state_synchronize_rcu);

/**
 * cond_synchronize_rcu - Conditionally wait for an RCU grace period
 *
 * @oldstate: return value from earlier call to get_state_synchronize_rcu()
 *
 * If a full RCU grace period has elapsed since the earlier call to
 * get_state_synchronize_rcu(), just return.  Otherwise, invoke
 * synchronize_rcu() to wait for a full grace period.
 *
 * Yes, this function does not take counter wrap into account.  But
 * counter wrap is harmless.  If the counter wraps, we have waited for
 * more than 2 billion grace periods (and way more on a 64-bit system!),
 * so waiting for one additional grace period should be just fine.
 */
void cond_synchronize_rcu(unsigned long oldstate)
{
	if (!rcu_seq_done(&rcu_state_p->gp_seq, oldstate))
		synchronize_rcu();
	else
		smp_mb(); /* Ensure GP ends before subsequent accesses. */
}
EXPORT_SYMBOL_GPL(cond_synchronize_rcu);

/**
 * get_state_synchronize_sched - Snapshot current RCU-sched state
 *
 * Returns a cookie that is used by a later call to cond_synchronize_sched()
 * to determine whether or not a full grace period has elapsed in the
 * meantime.
 */
static int rcu_pending(int user)
{
	bool gp_in_progress;
	struct rcu_data *rdp = this_cpu_ptr(&rcu_data);
	struct rcu_node *rnp = rdp->mynode;

	/* Check for CPU stalls, if enabled. */
	check_cpu_stall(rsp, rdp);

	/* Is this a nohz_full CPU in userspace or idle?  (Ignore RCU if so.) */
	if ((user || rcu_is_cpu_rrupt_from_idle()) && rcu_nohz_full_cpu())
		return 0;

	/* Is the RCU core waiting for a quiescent state from this CPU? */
	gp_in_progress = rcu_gp_in_progress();
	if (rdp->core_needs_qs && !rdp->cpu_no_qs.b.norm && gp_in_progress)
		return 1;

	/* Does this CPU have callbacks ready to invoke? */
	if (rcu_segcblist_ready_cbs(&rdp->cblist))
		return 1;

	/* Has RCU gone idle with this CPU needing another grace period? */
	if (!gp_in_progress && rcu_segcblist_is_enabled(&rdp->cblist) &&
	    (!IS_ENABLED(CONFIG_RCU_NOCB_CPU) ||
	     !rcu_segcblist_is_offloaded(&rdp->cblist)) &&
	    !rcu_segcblist_restempty(&rdp->cblist, RCU_NEXT_READY_TAIL))
		return 1;

	/* Have RCU grace period completed or started?  */
	if (rcu_seq_current(&rnp->gp_seq) != rdp->gp_seq ||
	    unlikely(READ_ONCE(rdp->gpwrap))) /* outside lock */
		return 1;

	/* Does this CPU need a deferred NOCB wakeup? */
	if (rcu_nocb_need_deferred_wakeup(rdp))
		return 1;

	/* nothing to do */
	return 0;
}

/*
 * Check to see if there is any immediate RCU-related work to be done
 * by the current CPU, returning 1 if so.  This function is part of the
 * RCU implementation; it is -not- an exported member of the RCU API.
 */
static int rcu_pending(void)
{
	struct rcu_state *rsp;

	for_each_rcu_flavor(rsp)
		if (__rcu_pending(rsp, this_cpu_ptr(rsp->rda)))
			return 1;
	return 0;
}

/*
 * RCU callback function for rcu_barrier().  If we are last, wake
 * up the task executing rcu_barrier().
 *
 * Note that the value of rcu_state.barrier_sequence must be captured
 * before the atomic_dec_and_test().  Otherwise, if this CPU is not last,
 * other CPUs might count the value down to zero before this CPU gets
 * around to invoking rcu_barrier_trace(), which might result in bogus
 * data from the next instance of rcu_barrier().
 */
static void rcu_barrier_callback(struct rcu_head *rhp)
{
	unsigned long __maybe_unused s = rcu_state.barrier_sequence;

	if (atomic_dec_and_test(&rcu_state.barrier_cpu_count)) {
		rcu_barrier_trace(TPS("LastCB"), -1, s);
		complete(&rcu_state.barrier_completion);
	} else {
		rcu_barrier_trace(TPS("CB"), -1, s);
	}
}

/*
 * Called with preemption disabled, and from cross-cpu IRQ context.
 */
static void rcu_barrier_func(void *cpu_in)
{
	uintptr_t cpu = (uintptr_t)cpu_in;
	struct rcu_data *rdp = per_cpu_ptr(&rcu_data, cpu);

	_rcu_barrier_trace(rsp, TPS("IRQ"), -1, rsp->barrier_sequence);
	rdp->barrier_head.func = rcu_barrier_callback;
	debug_rcu_head_queue(&rdp->barrier_head);
	rcu_nocb_lock(rdp);
	WARN_ON_ONCE(!rcu_nocb_flush_bypass(rdp, NULL, jiffies));
	if (rcu_segcblist_entrain(&rdp->cblist, &rdp->barrier_head)) {
		atomic_inc(&rcu_state.barrier_cpu_count);
	} else {
		debug_rcu_head_unqueue(&rdp->barrier_head);
		rcu_barrier_trace(TPS("IRQNQ"), -1,
				  rcu_state.barrier_sequence);
	}
}

/*
 * Orchestrate the specified type of RCU barrier, waiting for all
 * RCU callbacks of the specified type to complete.
 */
static void _rcu_barrier(struct rcu_state *rsp)
{
	uintptr_t cpu;
	struct rcu_data *rdp;
	unsigned long s = rcu_seq_snap(&rsp->barrier_sequence);

	_rcu_barrier_trace(rsp, TPS("Begin"), -1, s);

	/* Take mutex to serialize concurrent rcu_barrier() requests. */
	mutex_lock(&rsp->barrier_mutex);

	/* Did someone else do our work for us? */
	if (rcu_seq_done(&rcu_state.barrier_sequence, s)) {
		rcu_barrier_trace(TPS("EarlyExit"), -1,
				  rcu_state.barrier_sequence);
		smp_mb(); /* caller's subsequent code after above check. */
		mutex_unlock(&rsp->barrier_mutex);
		return;
	}

	/* Mark the start of the barrier operation. */
	rcu_seq_start(&rsp->barrier_sequence);
	_rcu_barrier_trace(rsp, TPS("Inc1"), -1, rsp->barrier_sequence);

	/*
	 * Initialize the count to two rather than to zero in order
	 * to avoid a too-soon return to zero in case of an immediate
	 * invocation of the just-enqueued callback (or preemption of
	 * this task).  Exclude CPU-hotplug operations to ensure that no
	 * offline non-offloaded CPU has callbacks queued.
	 */
	init_completion(&rcu_state.barrier_completion);
	atomic_set(&rcu_state.barrier_cpu_count, 2);
	get_online_cpus();

	/*
	 * Force each CPU with callbacks to register a new callback.
	 * When that callback is invoked, we will know that all of the
	 * corresponding CPU's preceding callbacks have been invoked.
	 */
	for_each_possible_cpu(cpu) {
		rdp = per_cpu_ptr(&rcu_data, cpu);
		if (cpu_is_offline(cpu) &&
		    !rcu_segcblist_is_offloaded(&rdp->cblist))
			continue;
		if (rcu_segcblist_n_cbs(&rdp->cblist) && cpu_online(cpu)) {
			rcu_barrier_trace(TPS("OnlineQ"), cpu,
					  rcu_state.barrier_sequence);
			smp_call_function_single(cpu, rcu_barrier_func, (void *)cpu, 1);
		} else if (rcu_segcblist_n_cbs(&rdp->cblist) &&
			   cpu_is_offline(cpu)) {
			rcu_barrier_trace(TPS("OfflineNoCBQ"), cpu,
					  rcu_state.barrier_sequence);
			local_irq_disable();
			rcu_barrier_func((void *)cpu);
			local_irq_enable();
		} else if (cpu_is_offline(cpu)) {
			rcu_barrier_trace(TPS("OfflineNoCBNoQ"), cpu,
					  rcu_state.barrier_sequence);
		} else {
			rcu_barrier_trace(TPS("OnlineNQ"), cpu,
					  rcu_state.barrier_sequence);
		}
	}
	put_online_cpus();

	/*
	 * Now that we have an rcu_barrier_callback() callback on each
	 * CPU, and thus each counted, remove the initial count.
	 */
	if (atomic_sub_and_test(2, &rcu_state.barrier_cpu_count))
		complete(&rcu_state.barrier_completion);

	/* Wait for all rcu_barrier_callback() callbacks to be invoked. */
	wait_for_completion(&rsp->barrier_completion);

	/* Mark the end of the barrier operation. */
	_rcu_barrier_trace(rsp, TPS("Inc2"), -1, rsp->barrier_sequence);
	rcu_seq_end(&rsp->barrier_sequence);

	/* Other rcu_barrier() invocations can now safely proceed. */
	mutex_unlock(&rsp->barrier_mutex);
}

/**
 * rcu_barrier_bh - Wait until all in-flight call_rcu_bh() callbacks complete.
 */
void rcu_barrier_bh(void)
{
	_rcu_barrier(&rcu_bh_state);
}
EXPORT_SYMBOL_GPL(rcu_barrier_bh);

/**
 * rcu_barrier_sched - Wait for in-flight call_rcu_sched() callbacks.
 */
void rcu_barrier_sched(void)
{
	_rcu_barrier(&rcu_sched_state);
}
EXPORT_SYMBOL_GPL(rcu_barrier_sched);

/*
 * Propagate ->qsinitmask bits up the rcu_node tree to account for the
 * first CPU in a given leaf rcu_node structure coming online.  The caller
 * must hold the corresponding leaf rcu_node ->lock with interrrupts
 * disabled.
 */
static void rcu_init_new_rnp(struct rcu_node *rnp_leaf)
{
	long mask;
	long oldmask;
	struct rcu_node *rnp = rnp_leaf;

	raw_lockdep_assert_held_rcu_node(rnp_leaf);
	WARN_ON_ONCE(rnp->wait_blkd_tasks);
	for (;;) {
		mask = rnp->grpmask;
		rnp = rnp->parent;
		if (rnp == NULL)
			return;
		raw_spin_lock_rcu_node(rnp); /* Interrupts already disabled. */
		oldmask = rnp->qsmaskinit;
		rnp->qsmaskinit |= mask;
		raw_spin_unlock_rcu_node(rnp); /* Interrupts remain disabled. */
		if (oldmask)
			return;
	}
}

/*
 * Do boot-time initialization of a CPU's per-CPU RCU data.
 */
static void __init
rcu_boot_init_percpu_data(int cpu, struct rcu_state *rsp)
{
	struct rcu_data *rdp = per_cpu_ptr(rsp->rda, cpu);

	/* Set up local state, ensuring consistent view of global state. */
	rdp->grpmask = leaf_node_cpu_bit(rdp->mynode, cpu);
	rdp->dynticks = &per_cpu(rcu_dynticks, cpu);
	WARN_ON_ONCE(rdp->dynticks->dynticks_nesting != 1);
	WARN_ON_ONCE(rcu_dynticks_in_eqs(rcu_dynticks_snap(rdp->dynticks)));
	rdp->rcu_ofl_gp_seq = rsp->gp_seq;
	rdp->rcu_ofl_gp_flags = RCU_GP_CLEANED;
	rdp->rcu_onl_gp_seq = rsp->gp_seq;
	rdp->rcu_onl_gp_flags = RCU_GP_CLEANED;
	rdp->cpu = cpu;
	rdp->rsp = rsp;
	rcu_boot_init_nocb_percpu_data(rdp);
}

/*
 * Initialize a CPU's per-CPU RCU data.  Note that only one online or
 * offline event can be happening at a given time.  Note also that we can
 * accept some slop in the rsp->gp_seq access due to the fact that this
 * CPU cannot possibly have any RCU callbacks in flight yet.
 */
static void
rcu_init_percpu_data(int cpu, struct rcu_state *rsp)
{
	unsigned long flags;
	struct rcu_data *rdp = per_cpu_ptr(rsp->rda, cpu);
	struct rcu_node *rnp = rcu_get_root(rsp);

	/* Set up local state, ensuring consistent view of global state. */
	raw_spin_lock_irqsave_rcu_node(rnp, flags);
	rdp->qlen_last_fqs_check = 0;
	rdp->n_force_qs_snap = rsp->n_force_qs;
	rdp->blimit = blimit;
	if (rcu_segcblist_empty(&rdp->cblist) && /* No early-boot CBs? */
	    !init_nocb_callback_list(rdp))
		rcu_segcblist_init(&rdp->cblist);  /* Re-enable callbacks. */
	rdp->dynticks->dynticks_nesting = 1;	/* CPU not up, no tearing. */
	rcu_dynticks_eqs_online();
	raw_spin_unlock_rcu_node(rnp);		/* irqs remain disabled. */

	/*
	 * Add CPU to leaf rcu_node pending-online bitmask.  Any needed
	 * propagation up the rcu_node tree will happen at the beginning
	 * of the next grace period.
	 */
	rnp = rdp->mynode;
	raw_spin_lock_rcu_node(rnp);		/* irqs already disabled. */
	rdp->beenonline = true;	 /* We have now been online. */
	rdp->gp_seq = READ_ONCE(rnp->gp_seq);
	rdp->gp_seq_needed = rdp->gp_seq;
	rdp->cpu_no_qs.b.norm = true;
	rdp->rcu_qs_ctr_snap = per_cpu(rcu_dynticks.rcu_qs_ctr, cpu);
	rdp->core_needs_qs = false;
	rdp->rcu_iw_pending = false;
	rdp->rcu_iw_gp_seq = rdp->gp_seq - 1;
	trace_rcu_grace_period(rcu_state.name, rdp->gp_seq, TPS("cpuonl"));
	raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
}

/*
 * Invoked early in the CPU-online process, when pretty much all
 * services are available.  The incoming CPU is not present.
 */
int rcutree_prepare_cpu(unsigned int cpu)
{
	struct rcu_state *rsp;

	for_each_rcu_flavor(rsp)
		rcu_init_percpu_data(cpu, rsp);

	rcu_prepare_kthreads(cpu);
	rcu_spawn_all_nocb_kthreads(cpu);

	return 0;
}

/*
 * Update RCU priority boot kthread affinity for CPU-hotplug changes.
 */
static void rcutree_affinity_setting(unsigned int cpu, int outgoing)
{
	struct rcu_data *rdp = per_cpu_ptr(rcu_state_p->rda, cpu);

	rcu_boost_kthread_setaffinity(rdp->mynode, outgoing);
}

/*
 * Near the end of the CPU-online process.  Pretty much all services
 * enabled, and the CPU is now very much alive.
 */
int rcutree_online_cpu(unsigned int cpu)
{
	unsigned long flags;
	struct rcu_data *rdp;
	struct rcu_node *rnp;
	struct rcu_state *rsp;

	for_each_rcu_flavor(rsp) {
		rdp = per_cpu_ptr(rsp->rda, cpu);
		rnp = rdp->mynode;
		raw_spin_lock_irqsave_rcu_node(rnp, flags);
		rnp->ffmask |= rdp->grpmask;
		raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
	}
	if (IS_ENABLED(CONFIG_TREE_SRCU))
		srcu_online_cpu(cpu);
	if (rcu_scheduler_active == RCU_SCHEDULER_INACTIVE)
		return 0; /* Too early in boot for scheduler work. */
	sync_sched_exp_online_cleanup(cpu);
	rcutree_affinity_setting(cpu, -1);

	// Stop-machine done, so allow nohz_full to disable tick.
	tick_dep_clear(TICK_DEP_BIT_RCU);
	return 0;
}

/*
 * Near the beginning of the process.  The CPU is still very much alive
 * with pretty much all services enabled.
 */
int rcutree_offline_cpu(unsigned int cpu)
{
	unsigned long flags;
	struct rcu_data *rdp;
	struct rcu_node *rnp;
	struct rcu_state *rsp;

	for_each_rcu_flavor(rsp) {
		rdp = per_cpu_ptr(rsp->rda, cpu);
		rnp = rdp->mynode;
		raw_spin_lock_irqsave_rcu_node(rnp, flags);
		rnp->ffmask &= ~rdp->grpmask;
		raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
	}

	rcutree_affinity_setting(cpu, cpu);

	// nohz_full CPUs need the tick for stop-machine to work quickly
	tick_dep_set(TICK_DEP_BIT_RCU);
	return 0;
}

static DEFINE_PER_CPU(int, rcu_cpu_started);

/*
 * Mark the specified CPU as being online so that subsequent grace periods
 * (both expedited and normal) will wait on it.  Note that this means that
 * incoming CPUs are not allowed to use RCU read-side critical sections
 * until this function is called.  Failing to observe this restriction
 * will result in lockdep splats.
 *
 * Note that this function is special in that it is invoked directly
 * from the incoming CPU rather than from the cpuhp_step mechanism.
 * This is because this function must be invoked at a precise location.
 */
void rcu_cpu_starting(unsigned int cpu)
{
	unsigned long flags;
	unsigned long mask;
	struct rcu_data *rdp;
	struct rcu_node *rnp;
	bool newcpu;

	if (per_cpu(rcu_cpu_started, cpu))
		return;

	per_cpu(rcu_cpu_started, cpu) = 1;

	rdp = per_cpu_ptr(&rcu_data, cpu);
	rnp = rdp->mynode;
	mask = rdp->grpmask;
	raw_spin_lock_irqsave_rcu_node(rnp, flags);
	WRITE_ONCE(rnp->qsmaskinitnext, rnp->qsmaskinitnext | mask);
	newcpu = !(rnp->expmaskinitnext & mask);
	rnp->expmaskinitnext |= mask;
	/* Allow lockless access for expedited grace periods. */
	smp_store_release(&rcu_state.ncpus, rcu_state.ncpus + newcpu); /* ^^^ */
	ASSERT_EXCLUSIVE_WRITER(rcu_state.ncpus);
	rcu_gpnum_ovf(rnp, rdp); /* Offline-induced counter wrap? */
	rdp->rcu_onl_gp_seq = READ_ONCE(rcu_state.gp_seq);
	rdp->rcu_onl_gp_flags = READ_ONCE(rcu_state.gp_flags);
	if (rnp->qsmask & mask) { /* RCU waiting on incoming CPU? */
		rcu_disable_urgency_upon_qs(rdp);
		/* Report QS -after- changing ->qsmaskinitnext! */
		rcu_report_qs_rnp(mask, rnp, rnp->gp_seq, flags);
	} else {
		raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
	}
	smp_mb(); /* Ensure RCU read-side usage follows above initialization. */
}

#ifdef CONFIG_HOTPLUG_CPU
/*
 * The CPU is exiting the idle loop into the arch_cpu_idle_dead()
 * function.  We now remove it from the rcu_node tree's ->qsmaskinitnext
 * bit masks.
 */
static void rcu_cleanup_dying_idle_cpu(int cpu, struct rcu_state *rsp)
{
	unsigned long flags;
	unsigned long mask;
	struct rcu_data *rdp = per_cpu_ptr(rsp->rda, cpu);
	struct rcu_node *rnp = rdp->mynode;  /* Outgoing CPU's rdp & rnp. */

	/* Remove outgoing CPU from mask in the leaf rcu_node structure. */
	mask = rdp->grpmask;
	spin_lock(&rsp->ofl_lock);
	raw_spin_lock_irqsave_rcu_node(rnp, flags); /* Enforce GP memory-order guarantee. */
	rdp->rcu_ofl_gp_seq = READ_ONCE(rsp->gp_seq);
	rdp->rcu_ofl_gp_flags = READ_ONCE(rsp->gp_flags);
	if (rnp->qsmask & mask) { /* RCU waiting on outgoing CPU? */
		/* Report quiescent state -before- changing ->qsmaskinitnext! */
		rcu_report_qs_rnp(mask, rsp, rnp, rnp->gp_seq, flags);
		raw_spin_lock_irqsave_rcu_node(rnp, flags);
	}
	rnp->qsmaskinitnext &= ~mask;
	raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
	spin_unlock(&rsp->ofl_lock);
}

/*
 * The outgoing function has no further need of RCU, so remove it from
 * the list of CPUs that RCU must track.
 *
 * Note that this function is special in that it is invoked directly
 * from the outgoing CPU rather than from the cpuhp_step mechanism.
 * This is because this function must be invoked at a precise location.
 */
void rcu_report_dead(unsigned int cpu)
{
	struct rcu_state *rsp;

	/* QS for any half-done expedited RCU-sched GP. */
	preempt_disable();
	rcu_report_exp_rdp(&rcu_sched_state,
			   this_cpu_ptr(rcu_sched_state.rda), true);
	preempt_enable();
	rcu_preempt_deferred_qs(current);

	/* Remove outgoing CPU from mask in the leaf rcu_node structure. */
	mask = rdp->grpmask;
	raw_spin_lock(&rcu_state.ofl_lock);
	raw_spin_lock_irqsave_rcu_node(rnp, flags); /* Enforce GP memory-order guarantee. */
	rdp->rcu_ofl_gp_seq = READ_ONCE(rcu_state.gp_seq);
	rdp->rcu_ofl_gp_flags = READ_ONCE(rcu_state.gp_flags);
	if (rnp->qsmask & mask) { /* RCU waiting on outgoing CPU? */
		/* Report quiescent state -before- changing ->qsmaskinitnext! */
		rcu_report_qs_rnp(mask, rnp, rnp->gp_seq, flags);
		raw_spin_lock_irqsave_rcu_node(rnp, flags);
	}
	WRITE_ONCE(rnp->qsmaskinitnext, rnp->qsmaskinitnext & ~mask);
	raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
	raw_spin_unlock(&rcu_state.ofl_lock);

	per_cpu(rcu_cpu_started, cpu) = 0;
}

/* Migrate the dead CPU's callbacks to the current CPU. */
static void rcu_migrate_callbacks(int cpu, struct rcu_state *rsp)
{
	unsigned long flags;
	struct rcu_data *my_rdp;
	struct rcu_data *rdp = per_cpu_ptr(rsp->rda, cpu);
	struct rcu_node *rnp_root = rcu_get_root(rdp->rsp);
	bool needwake;

	if (rcu_is_nocb_cpu(cpu) || rcu_segcblist_empty(&rdp->cblist))
		return;  /* No callbacks to migrate. */

	local_irq_save(flags);
	my_rdp = this_cpu_ptr(rsp->rda);
	if (rcu_nocb_adopt_orphan_cbs(my_rdp, rdp, flags)) {
		local_irq_restore(flags);
		return;
	}
	raw_spin_lock_rcu_node(rnp_root); /* irqs already disabled. */
	/* Leverage recent GPs and set GP for new callbacks. */
	needwake = rcu_advance_cbs(rsp, rnp_root, rdp) ||
		   rcu_advance_cbs(rsp, rnp_root, my_rdp);
	rcu_segcblist_merge(&my_rdp->cblist, &rdp->cblist);
	WARN_ON_ONCE(rcu_segcblist_empty(&my_rdp->cblist) !=
		     !rcu_segcblist_n_cbs(&my_rdp->cblist));
	raw_spin_unlock_irqrestore_rcu_node(rnp_root, flags);
	if (needwake)
		rcu_gp_kthread_wake(rsp);
	WARN_ONCE(rcu_segcblist_n_cbs(&rdp->cblist) != 0 ||
		  !rcu_segcblist_empty(&rdp->cblist),
		  "rcu_cleanup_dead_cpu: Callbacks on offline CPU %d: qlen=%lu, 1stCB=%p\n",
		  cpu, rcu_segcblist_n_cbs(&rdp->cblist),
		  rcu_segcblist_first_cb(&rdp->cblist));
}

/*
 * The outgoing CPU has just passed through the dying-idle state,
 * and we are being invoked from the CPU that was IPIed to continue the
 * offline operation.  We need to migrate the outgoing CPU's callbacks.
 */
void rcutree_migrate_callbacks(int cpu)
{
	struct rcu_state *rsp;

	for_each_rcu_flavor(rsp)
		rcu_migrate_callbacks(cpu, rsp);
}
#endif

/*
 * On non-huge systems, use expedited RCU grace periods to make suspend
 * and hibernation run faster.
 */
static int rcu_pm_notify(struct notifier_block *self,
			 unsigned long action, void *hcpu)
{
	switch (action) {
	case PM_HIBERNATION_PREPARE:
	case PM_SUSPEND_PREPARE:
		if (nr_cpu_ids <= 256) /* Expediting bad for large systems. */
			rcu_expedite_gp();
		break;
	case PM_POST_HIBERNATION:
	case PM_POST_SUSPEND:
		if (nr_cpu_ids <= 256) /* Expediting bad for large systems. */
			rcu_unexpedite_gp();
		break;
	default:
		break;
	}
	return NOTIFY_OK;
}

/*
 * Spawn the kthreads that handle each RCU flavor's grace periods.
 */
static int __init rcu_spawn_gp_kthread(void)
{
	unsigned long flags;
	int kthread_prio_in = kthread_prio;
	struct rcu_node *rnp;
	struct rcu_state *rsp;
	struct sched_param sp;
	struct task_struct *t;

	/* Force priority into range. */
	if (IS_ENABLED(CONFIG_RCU_BOOST) && kthread_prio < 2
	    && IS_BUILTIN(CONFIG_RCU_TORTURE_TEST))
		kthread_prio = 2;
	else if (IS_ENABLED(CONFIG_RCU_BOOST) && kthread_prio < 1)
		kthread_prio = 1;
	else if (kthread_prio < 0)
		kthread_prio = 0;
	else if (kthread_prio > 99)
		kthread_prio = 99;

	if (kthread_prio != kthread_prio_in)
		pr_alert("rcu_spawn_gp_kthread(): Limited prio to %d from %d\n",
			 kthread_prio, kthread_prio_in);

	rcu_scheduler_fully_active = 1;
	for_each_rcu_flavor(rsp) {
		t = kthread_create(rcu_gp_kthread, rsp, "%s", rsp->name);
		BUG_ON(IS_ERR(t));
		rnp = rcu_get_root(rsp);
		raw_spin_lock_irqsave_rcu_node(rnp, flags);
		rsp->gp_kthread = t;
		if (kthread_prio) {
			sp.sched_priority = kthread_prio;
			sched_setscheduler_nocheck(t, SCHED_RR, &sp);
		}
		raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
		wake_up_process(t);
	}
	rnp = rcu_get_root();
	raw_spin_lock_irqsave_rcu_node(rnp, flags);
	WRITE_ONCE(rcu_state.gp_activity, jiffies);
	WRITE_ONCE(rcu_state.gp_req_activity, jiffies);
	// Reset .gp_activity and .gp_req_activity before setting .gp_kthread.
	smp_store_release(&rcu_state.gp_kthread, t);  /* ^^^ */
	raw_spin_unlock_irqrestore_rcu_node(rnp, flags);
	wake_up_process(t);
	rcu_spawn_nocb_kthreads();
	rcu_spawn_boost_kthreads();
	return 0;
}
early_initcall(rcu_spawn_gp_kthread);

/*
 * This function is invoked towards the end of the scheduler's
 * initialization process.  Before this is called, the idle task might
 * contain synchronous grace-period primitives (during which time, this idle
 * task is booting the system, and such primitives are no-ops).  After this
 * function is called, any synchronous grace-period primitives are run as
 * expedited, with the requesting task driving the grace period forward.
 * A later core_initcall() rcu_set_runtime_mode() will switch to full
 * runtime RCU functionality.
 */
void rcu_scheduler_starting(void)
{
	WARN_ON(num_online_cpus() != 1);
	WARN_ON(nr_context_switches() > 0);
	rcu_test_sync_prims();
	rcu_scheduler_active = RCU_SCHEDULER_INIT;
	rcu_test_sync_prims();
}

/*
 * Helper function for rcu_init() that initializes one rcu_state structure.
 */
static void __init rcu_init_one(struct rcu_state *rsp)
{
	static const char * const buf[] = RCU_NODE_NAME_INIT;
	static const char * const fqs[] = RCU_FQS_NAME_INIT;
	static struct lock_class_key rcu_node_class[RCU_NUM_LVLS];
	static struct lock_class_key rcu_fqs_class[RCU_NUM_LVLS];

	int levelspread[RCU_NUM_LVLS];		/* kids/node in each level. */
	int cpustride = 1;
	int i;
	int j;
	struct rcu_node *rnp;

	BUILD_BUG_ON(RCU_NUM_LVLS > ARRAY_SIZE(buf));  /* Fix buf[] init! */

	/* Silence gcc 4.8 false positive about array index out of range. */
	if (rcu_num_lvls <= 0 || rcu_num_lvls > RCU_NUM_LVLS)
		panic("rcu_init_one: rcu_num_lvls out of range");

	/* Initialize the level-tracking arrays. */

	for (i = 1; i < rcu_num_lvls; i++)
		rsp->level[i] = rsp->level[i - 1] + num_rcu_lvl[i - 1];
	rcu_init_levelspread(levelspread, num_rcu_lvl);

	/* Initialize the elements themselves, starting from the leaves. */

	for (i = rcu_num_lvls - 1; i >= 0; i--) {
		cpustride *= levelspread[i];
		rnp = rsp->level[i];
		for (j = 0; j < num_rcu_lvl[i]; j++, rnp++) {
			raw_spin_lock_init(&ACCESS_PRIVATE(rnp, lock));
			lockdep_set_class_and_name(&ACCESS_PRIVATE(rnp, lock),
						   &rcu_node_class[i], buf[i]);
			raw_spin_lock_init(&rnp->fqslock);
			lockdep_set_class_and_name(&rnp->fqslock,
						   &rcu_fqs_class[i], fqs[i]);
			rnp->gp_seq = rsp->gp_seq;
			rnp->gp_seq_needed = rsp->gp_seq;
			rnp->completedqs = rsp->gp_seq;
			rnp->qsmask = 0;
			rnp->qsmaskinit = 0;
			rnp->grplo = j * cpustride;
			rnp->grphi = (j + 1) * cpustride - 1;
			if (rnp->grphi >= nr_cpu_ids)
				rnp->grphi = nr_cpu_ids - 1;
			if (i == 0) {
				rnp->grpnum = 0;
				rnp->grpmask = 0;
				rnp->parent = NULL;
			} else {
				rnp->grpnum = j % levelspread[i - 1];
				rnp->grpmask = 1UL << rnp->grpnum;
				rnp->parent = rsp->level[i - 1] +
					      j / levelspread[i - 1];
			}
			rnp->level = i;
			INIT_LIST_HEAD(&rnp->blkd_tasks);
			rcu_init_one_nocb(rnp);
			init_waitqueue_head(&rnp->exp_wq[0]);
			init_waitqueue_head(&rnp->exp_wq[1]);
			init_waitqueue_head(&rnp->exp_wq[2]);
			init_waitqueue_head(&rnp->exp_wq[3]);
			spin_lock_init(&rnp->exp_lock);
		}
	}

	init_swait_queue_head(&rsp->gp_wq);
	init_swait_queue_head(&rsp->expedited_wq);
	rnp = rcu_first_leaf_node(rsp);
	for_each_possible_cpu(i) {
		while (i > rnp->grphi)
			rnp++;
		per_cpu_ptr(rsp->rda, i)->mynode = rnp;
		rcu_boot_init_percpu_data(i, rsp);
	}
	list_add(&rsp->flavors, &rcu_struct_flavors);
}

/*
 * Compute the rcu_node tree geometry from kernel parameters.  This cannot
 * replace the definitions in tree.h because those are needed to size
 * the ->node array in the rcu_state structure.
 */
static void __init rcu_init_geometry(void)
{
	ulong d;
	int i;
	int rcu_capacity[RCU_NUM_LVLS];

	/*
	 * Initialize any unspecified boot parameters.
	 * The default values of jiffies_till_first_fqs and
	 * jiffies_till_next_fqs are set to the RCU_JIFFIES_TILL_FORCE_QS
	 * value, which is a function of HZ, then adding one for each
	 * RCU_JIFFIES_FQS_DIV CPUs that might be on the system.
	 */
	d = RCU_JIFFIES_TILL_FORCE_QS + nr_cpu_ids / RCU_JIFFIES_FQS_DIV;
	if (jiffies_till_first_fqs == ULONG_MAX)
		jiffies_till_first_fqs = d;
	if (jiffies_till_next_fqs == ULONG_MAX)
		jiffies_till_next_fqs = d;

	/* If the compile-time values are accurate, just leave. */
	if (rcu_fanout_leaf == RCU_FANOUT_LEAF &&
	    nr_cpu_ids == NR_CPUS)
		return;
	pr_info("Adjusting geometry for rcu_fanout_leaf=%d, nr_cpu_ids=%u\n",
		rcu_fanout_leaf, nr_cpu_ids);

	/*
	 * The boot-time rcu_fanout_leaf parameter must be at least two
	 * and cannot exceed the number of bits in the rcu_node masks.
	 * Complain and fall back to the compile-time values if this
	 * limit is exceeded.
	 */
	if (rcu_fanout_leaf < 2 ||
	    rcu_fanout_leaf > sizeof(unsigned long) * 8) {
		rcu_fanout_leaf = RCU_FANOUT_LEAF;
		WARN_ON(1);
		return;
	}

	/*
	 * Compute number of nodes that can be handled an rcu_node tree
	 * with the given number of levels.
	 */
	rcu_capacity[0] = rcu_fanout_leaf;
	for (i = 1; i < RCU_NUM_LVLS; i++)
		rcu_capacity[i] = rcu_capacity[i - 1] * RCU_FANOUT;

	/*
	 * The tree must be able to accommodate the configured number of CPUs.
	 * If this limit is exceeded, fall back to the compile-time values.
	 */
	if (nr_cpu_ids > rcu_capacity[RCU_NUM_LVLS - 1]) {
		rcu_fanout_leaf = RCU_FANOUT_LEAF;
		WARN_ON(1);
		return;
	}

	/* Calculate the number of levels in the tree. */
	for (i = 0; nr_cpu_ids > rcu_capacity[i]; i++) {
	}
	rcu_num_lvls = i + 1;

	/* Calculate the number of rcu_nodes at each level of the tree. */
	for (i = 0; i < rcu_num_lvls; i++) {
		int cap = rcu_capacity[(rcu_num_lvls - 1) - i];
		num_rcu_lvl[i] = DIV_ROUND_UP(nr_cpu_ids, cap);
	}

	/* Calculate the total number of rcu_node structures. */
	rcu_num_nodes = 0;
	for (i = 0; i < rcu_num_lvls; i++)
		rcu_num_nodes += num_rcu_lvl[i];
}

/*
 * Dump out the structure of the rcu_node combining tree associated
 * with the rcu_state structure referenced by rsp.
 */
static void __init rcu_dump_rcu_node_tree(struct rcu_state *rsp)
{
	int level = 0;
	struct rcu_node *rnp;

	pr_info("rcu_node tree layout dump\n");
	pr_info(" ");
	rcu_for_each_node_breadth_first(rsp, rnp) {
		if (rnp->level != level) {
			pr_cont("\n");
			pr_info(" ");
			level = rnp->level;
		}
		pr_cont("%d:%d ^%d  ", rnp->grplo, rnp->grphi, rnp->grpnum);
	}
	pr_cont("\n");
}

struct workqueue_struct *rcu_gp_wq;
struct workqueue_struct *rcu_par_gp_wq;

static void __init kfree_rcu_batch_init(void)
{
	int cpu;
	int i;

	for_each_possible_cpu(cpu) {
		struct kfree_rcu_cpu *krcp = per_cpu_ptr(&krc, cpu);
		struct kfree_rcu_bulk_data *bnode;

		for (i = 0; i < KFREE_N_BATCHES; i++) {
			INIT_RCU_WORK(&krcp->krw_arr[i].rcu_work, kfree_rcu_work);
			krcp->krw_arr[i].krcp = krcp;
		}

		for (i = 0; i < rcu_min_cached_objs; i++) {
			bnode = (struct kfree_rcu_bulk_data *)
				__get_free_page(GFP_NOWAIT | __GFP_NOWARN);

			if (bnode)
				put_cached_bnode(krcp, bnode);
			else
				pr_err("Failed to preallocate for %d CPU!\n", cpu);
		}

		INIT_DELAYED_WORK(&krcp->monitor_work, kfree_rcu_monitor);
		krcp->initialized = true;
	}
	if (register_shrinker(&kfree_rcu_shrinker))
		pr_err("Failed to register kfree_rcu() shrinker!\n");
}

void __init rcu_init(void)
{
	int cpu;

	rcu_early_boot_tests();

	kfree_rcu_batch_init();
	rcu_bootup_announce();
	rcu_init_geometry();
	rcu_init_one(&rcu_bh_state);
	rcu_init_one(&rcu_sched_state);
	if (dump_tree)
		rcu_dump_rcu_node_tree(&rcu_sched_state);
	__rcu_init_preempt();
	open_softirq(RCU_SOFTIRQ, rcu_process_callbacks);

	/*
	 * We don't need protection against CPU-hotplug here because
	 * this is called early in boot, before either interrupts
	 * or the scheduler are operational.
	 */
	pm_notifier(rcu_pm_notify, 0);
	for_each_online_cpu(cpu) {
		rcutree_prepare_cpu(cpu);
		rcu_cpu_starting(cpu);
		rcutree_online_cpu(cpu);
	}

	/* Create workqueue for expedited GPs and for Tree SRCU. */
	rcu_gp_wq = alloc_workqueue("rcu_gp", WQ_MEM_RECLAIM, 0);
	WARN_ON(!rcu_gp_wq);
	rcu_par_gp_wq = alloc_workqueue("rcu_par_gp", WQ_MEM_RECLAIM, 0);
	WARN_ON(!rcu_par_gp_wq);
	srcu_init();

	/* Fill in default value for rcutree.qovld boot parameter. */
	/* -After- the rcu_node ->lock fields are initialized! */
	if (qovld < 0)
		qovld_calc = DEFAULT_RCU_QOVLD_MULT * qhimark;
	else
		qovld_calc = qovld;
}

#include "tree_exp.h"
#include "tree_plugin.h"
