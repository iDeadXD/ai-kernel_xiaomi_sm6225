/* SPDX-License-Identifier: GPL-2.0 */
/* Common definitions for cpuidle governors. */
#ifndef __CPUIDLE_GOVERNOR_H
#define __CPUIDLE_GOVERNOR_H
/*
 * Idle state target residency threshold used for deciding whether or not to
 * check the time till the closest expected timer event.
 */
#define RESIDENCY_THRESHOLD_US	15
#endif /* __CPUIDLE_GOVERNOR_H */