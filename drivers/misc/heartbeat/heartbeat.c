//SPDX-License-Identifier: GPL-2.0-only
//Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.

#include<linux/init.h>
#include<linux/module.h>
#include<linux/kernel.h>
#include<linux/gpio.h>
#include<linux/of_gpio.h>
#include<linux/platform_device.h>
#include<linux/kobject.h>
#include<linux/sysfs.h>
#include<linux/workqueue.h>
#include<linux/mutex.h>

#define QTI_EVENT_TIMEOUT   3
#define HB_BUFFER_SIZE      1024

struct kobject *h_kobj;
uint32_t sysstate_value = 0;
struct delayed_work hwork;
struct mutex h_lock;
void *qti_can_priv_data = NULL;
char *hb_buf = NULL;

int send_heartbeat_event(void *, uint32_t, int);

void send_qti_events(struct work_struct *work)
{
	mutex_lock(&h_lock);
	send_heartbeat_event(qti_can_priv_data, sysstate_value, 4);
	mutex_unlock(&h_lock);
	sysstate_value = 0;
	schedule_delayed_work(&hwork, QTI_EVENT_TIMEOUT*HZ);
}

static ssize_t android_status_show(struct kobject *kobj,
		struct kobj_attribute *attr, char *buf)
{
	return snprintf(buf, HB_BUFFER_SIZE, "%s", hb_buf);
}

static ssize_t android_status_store(struct kobject *kobj,
		struct kobj_attribute *attr, const char *buf, size_t count)
{
	if (count >= HB_BUFFER_SIZE) {
		return -EINVAL;
	}
	strlcpy(hb_buf, buf, count);
	sscanf(buf, "%8x", &sysstate_value);
	mutex_lock(&h_lock);
	send_heartbeat_event(qti_can_priv_data, sysstate_value, 4);
	mutex_unlock(&h_lock);
	sysstate_value = 0;
	return count;
}

static struct kobj_attribute h_attr = __ATTR(sysstate_value, 0664, android_status_show, android_status_store);

int register_heartbeat(void *priv_data) {
	qti_can_priv_data = priv_data;
	return 0;
}
EXPORT_SYMBOL(register_heartbeat);

static int qti_heartbeat_probe(struct platform_device *pdev) {
	int ret = 0;

	h_kobj = kobject_create_and_add("qti_heartbeat",NULL);
	if (!h_kobj) {
		ret = -ENOMEM;
		return ret;
	}

	hb_buf = (char *)kzalloc(HB_BUFFER_SIZE, GFP_KERNEL);
	if (!hb_buf) {
		kobject_put(h_kobj);
		return -ENOMEM;
	}

	ret = sysfs_create_file(h_kobj, &h_attr.attr);
	if (ret) {
		kfree(hb_buf);
		kobject_put(h_kobj);
		pr_err("%s Failed to create /sys/qti_heartbeat/sysstate_value file\n",__func__);
		ret = -1;
		return ret;
	}

	mutex_init(&h_lock);

	INIT_DELAYED_WORK(&hwork, send_qti_events);
	schedule_delayed_work(&hwork, QTI_EVENT_TIMEOUT*HZ);
	pr_info("%s completed\n",__func__);
	return ret;
}

static int qti_heartbeat_remove(struct platform_device *pdev) {
	kfree(hb_buf);
	kobject_put(h_kobj);
	cancel_delayed_work_sync(&hwork);
	return 0;
}

static struct of_device_id qti_heartbeat_match_table[] = {
	{ .compatible = "qti,heartbeat" },
	{}
};
MODULE_DEVICE_TABLE(of, qti_heartbeat_match_table);

static struct platform_driver qti_heartbeat_platform_driver = {
	.probe = qti_heartbeat_probe,
	.remove = qti_heartbeat_remove,
	.driver = {
		.name = "qti-heartbeat",
		.owner = THIS_MODULE,
		.of_match_table = of_match_ptr(qti_heartbeat_match_table),
	},
};

module_platform_driver(qti_heartbeat_platform_driver);

MODULE_DESCRIPTION("qti heartbeat driver");
MODULE_LICENSE("GPL v2");
