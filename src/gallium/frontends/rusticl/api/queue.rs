// Copyright 2020 Red Hat.
// SPDX-License-Identifier: MIT

use crate::api::event::create_and_queue;
use crate::api::icd::*;
use crate::api::util::*;
use crate::core::context::*;
use crate::core::device::*;
use crate::core::event::*;
use crate::core::queue::*;

use mesa_rust_util::properties::*;
use rusticl_opencl_gen::*;
use rusticl_proc_macros::cl_entrypoint;
use rusticl_proc_macros::cl_info_entrypoint;

use std::ptr;
use std::sync::Arc;

#[cl_info_entrypoint(clGetCommandQueueInfo)]
unsafe impl CLInfo<cl_command_queue_info> for cl_command_queue {
    fn query(&self, q: cl_command_queue_info, v: CLInfoValue) -> CLResult<CLInfoRes> {
        let queue = Queue::ref_from_raw(*self)?;
        match q {
            CL_QUEUE_CONTEXT => {
                // Note we use as_ptr here which doesn't increase the reference count.
                let ptr = Arc::as_ptr(&queue.context);
                v.write::<cl_context>(cl_context::from_ptr(ptr))
            }
            CL_QUEUE_DEVICE => v.write::<cl_device_id>(cl_device_id::from_ptr(queue.device)),
            CL_QUEUE_DEVICE_DEFAULT => {
                // Note we use as_ptr here which doesn't increase the reference count.
                let def = queue.context.default_device_queue(queue.device);
                v.write::<cl_command_queue>(
                    def.map_or(ptr::null_mut(), |q| cl_command_queue::from_ptr(Arc::as_ptr(&q))),
                )
            }
            CL_QUEUE_PROPERTIES => v.write::<cl_command_queue_properties>(queue.props),
            CL_QUEUE_PROPERTIES_ARRAY => {
                v.write::<&Properties<cl_queue_properties>>(&queue.props_v2)
            }
            CL_QUEUE_REFERENCE_COUNT => v.write::<cl_uint>(Queue::refcnt(*self)?),
            // CL_INVALID_COMMAND_QUEUE for a host command-queue
            CL_QUEUE_SIZE => match &queue.dev_queue {
                Some(dq) => v.write::<cl_uint>(dq.size),
                None => Err(CL_INVALID_COMMAND_QUEUE),
            },
            // CL_INVALID_VALUE if param_name is not one of the supported values
            _ => Err(CL_INVALID_VALUE),
        }
    }
}

#[cl_entrypoint(clSetCommandQueueProperty)]
fn set_command_queue_property(
    _command_queue: cl_command_queue,
    _properties: cl_command_queue_properties,
    _enable: cl_bool,
    _old_properties: *mut cl_command_queue_properties,
) -> CLResult<()> {
    // clSetCommandQueueProperty may unconditionally return an error if no devices in the context
    // associated with command_queue support modifying the properties of a command-queue. Support
    // for modifying the properties of a command-queue is required only for OpenCL 1.0 devices.
    //
    // CL_INVALID_OPERATION if no devices in the context associated with command_queue support
    // modifying the properties of a command-queue.
    Err(CL_INVALID_OPERATION)
}

fn valid_command_queue_properties(properties: cl_command_queue_properties) -> bool {
    let valid_flags = cl_bitfield::from(
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
            | CL_QUEUE_PROFILING_ENABLE
            | CL_QUEUE_ON_DEVICE
            | CL_QUEUE_ON_DEVICE_DEFAULT,
    );
    properties & !valid_flags == 0
}

fn supported_command_queue_properties(
    dev: &Device,
    properties: cl_command_queue_properties,
) -> bool {
    let mut valid_flags = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    if dev.caps.has_timestamp {
        valid_flags |= CL_QUEUE_PROFILING_ENABLE;
    }

    if dev.device_enqueue_supported() {
        valid_flags |= CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT;
        // the host times the kernels of on-device queues (capture_event_profiling_info)
        if properties & cl_bitfield::from(CL_QUEUE_ON_DEVICE) != 0 {
            valid_flags |= CL_QUEUE_PROFILING_ENABLE;
        }
    }

    if properties & !cl_bitfield::from(valid_flags) != 0 {
        return false;
    }

    true
}

pub fn create_command_queue_impl(
    context: cl_context,
    device: cl_device_id,
    properties: cl_command_queue_properties,
    properties_v2: Properties<cl_queue_properties>,
    queue_size: Option<u32>,
) -> CLResult<cl_command_queue> {
    let c = Context::arc_from_raw(context)?;
    let d = Device::ref_from_raw(device)?
        .to_static()
        .ok_or(CL_INVALID_DEVICE)?;

    // CL_INVALID_DEVICE if device [...] is not associated with context.
    if !c.devs.contains(&d) {
        return Err(CL_INVALID_DEVICE);
    }

    // CL_INVALID_VALUE if values specified in properties are not valid.
    if !valid_command_queue_properties(properties) {
        return Err(CL_INVALID_VALUE);
    }

    // CL_INVALID_QUEUE_PROPERTIES if values specified in properties are valid but are not supported by the device.
    if !supported_command_queue_properties(d, properties) {
        return Err(CL_INVALID_QUEUE_PROPERTIES);
    }

    let on_device = bit_check(properties, CL_QUEUE_ON_DEVICE);
    // CL_INVALID_VALUE: an on-device queue is out of order; the default one is on the device; a
    // size only for an on-device queue, at most CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE
    if (on_device && !bit_check(properties, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE))
        || (!on_device && bit_check(properties, CL_QUEUE_ON_DEVICE_DEFAULT))
        || (!on_device && queue_size.is_some())
        || queue_size.is_some_and(|s| s > d.queue_on_device_max_size())
    {
        return Err(CL_INVALID_VALUE);
    }

    if !on_device {
        return Ok(Queue::new(c, d, properties, properties_v2, None)?.into_cl());
    }

    // one on-device queue per device: the default one (CL_DEVICE_MAX_ON_DEVICE_QUEUES 1)
    if let Some(q) = c.default_device_queue(d) {
        if bit_check(properties, CL_QUEUE_ON_DEVICE_DEFAULT) {
            return Ok(q.into_cl());
        }
        return Err(CL_OUT_OF_RESOURCES);
    }

    let size = queue_size.unwrap_or(d.queue_on_device_preferred_size());
    let dq = DeviceQueue::new(d, size)?;
    let q = Queue::new(Arc::clone(&c), d, properties, properties_v2, Some(dq))?;
    // CL_OUT_OF_RESOURCES beyond CL_DEVICE_MAX_ON_DEVICE_QUEUES
    if !c.add_device_queue(d, &q, 1) {
        return Err(CL_OUT_OF_RESOURCES);
    }
    if bit_check(properties, CL_QUEUE_ON_DEVICE_DEFAULT) {
        c.set_default_device_queue(d, &q);
    }
    Ok(q.into_cl())
}

#[cl_entrypoint(clCreateCommandQueue)]
fn create_command_queue(
    context: cl_context,
    device: cl_device_id,
    properties: cl_command_queue_properties,
) -> CLResult<cl_command_queue> {
    create_command_queue_impl(context, device, properties, Properties::default(), None)
}

#[cl_entrypoint(clCreateCommandQueueWithProperties)]
fn create_command_queue_with_properties(
    context: cl_context,
    device: cl_device_id,
    properties: *const cl_queue_properties,
) -> CLResult<cl_command_queue> {
    let mut queue_properties = cl_command_queue_properties::default();
    let mut queue_size = None;
    let d = Device::ref_from_raw(device)?;

    // SAFETY: properties is a 0 terminated array by spec.
    let properties = unsafe { Properties::new(properties) }.ok_or(CL_INVALID_VALUE)?;
    for (&key, &val) in properties.iter() {
        match u32::try_from(key).or(Err(CL_INVALID_VALUE))? {
            CL_QUEUE_PROPERTIES => queue_properties = val,
            CL_QUEUE_PRIORITY_KHR if d.context_priority_supported() != 0 => {
                let valid_props: cl_queue_properties = (CL_QUEUE_PRIORITY_LOW_KHR
                    | CL_QUEUE_PRIORITY_MED_KHR
                    | CL_QUEUE_PRIORITY_HIGH_KHR)
                    .into();

                if val & !valid_props != 0 {
                    return Err(CL_INVALID_VALUE);
                }
            }
            // CL_INVALID_QUEUE_PROPERTIES if values specified in properties are valid but are not
            // supported by the device.
            CL_QUEUE_SIZE if !d.device_enqueue_supported() => {
                return Err(CL_INVALID_QUEUE_PROPERTIES)
            }
            CL_QUEUE_SIZE => queue_size = Some(u32::try_from(val).or(Err(CL_INVALID_VALUE))?),
            _ => return Err(CL_INVALID_VALUE),
        }
    }

    create_command_queue_impl(context, device, queue_properties, properties, queue_size)
}

#[cl_entrypoint(clEnqueueMarker)]
fn enqueue_marker(command_queue: cl_command_queue, event: *mut cl_event) -> CLResult<()> {
    let q = Queue::arc_from_raw(command_queue)?;

    // TODO marker makes sure previous commands did complete
    create_and_queue(q, CL_COMMAND_MARKER, Vec::new(), event, false, None)
}

#[cl_entrypoint(clEnqueueMarkerWithWaitList)]
fn enqueue_marker_with_wait_list(
    command_queue: cl_command_queue,
    num_events_in_wait_list: cl_uint,
    event_wait_list: *const cl_event,
    event: *mut cl_event,
) -> CLResult<()> {
    let q = Queue::arc_from_raw(command_queue)?;
    let evs = event_list_from_cl(&q, num_events_in_wait_list, event_wait_list)?;

    // TODO marker makes sure previous commands did complete
    create_and_queue(q, CL_COMMAND_MARKER, evs, event, false, None)
}

#[cl_entrypoint(clEnqueueBarrier)]
fn enqueue_barrier(command_queue: cl_command_queue) -> CLResult<()> {
    let q = Queue::arc_from_raw(command_queue)?;

    // TODO barriers make sure previous commands did complete and other commands didn't start
    let e = Event::new(&q, CL_COMMAND_BARRIER, Vec::new(), None, None)?;
    q.queue(e);
    Ok(())
}

#[cl_entrypoint(clEnqueueBarrierWithWaitList)]
fn enqueue_barrier_with_wait_list(
    command_queue: cl_command_queue,
    num_events_in_wait_list: cl_uint,
    event_wait_list: *const cl_event,
    event: *mut cl_event,
) -> CLResult<()> {
    let q = Queue::arc_from_raw(command_queue)?;
    let evs = event_list_from_cl(&q, num_events_in_wait_list, event_wait_list)?;

    // TODO barriers make sure previous commands did complete and other commands didn't start
    create_and_queue(q, CL_COMMAND_BARRIER, evs, event, false, None)
}

#[cl_entrypoint(clFlush)]
fn flush(command_queue: cl_command_queue) -> CLResult<()> {
    // CL_INVALID_COMMAND_QUEUE if command_queue is not a valid host command-queue.
    let q = Queue::ref_from_raw(command_queue)?;
    if q.dev_queue.is_some() {
        return Err(CL_INVALID_COMMAND_QUEUE);
    }
    q.flush(false)
}

#[cl_entrypoint(clFinish)]
fn finish(command_queue: cl_command_queue) -> CLResult<()> {
    // CL_INVALID_COMMAND_QUEUE if command_queue is not a valid host command-queue.
    let q = Queue::ref_from_raw(command_queue)?;
    if q.dev_queue.is_some() {
        return Err(CL_INVALID_COMMAND_QUEUE);
    }
    q.flush(true)
}

#[cl_entrypoint(clRetainCommandQueue)]
fn retain_command_queue(command_queue: cl_command_queue) -> CLResult<()> {
    Queue::retain(command_queue)
}

#[cl_entrypoint(clReleaseCommandQueue)]
fn release_command_queue(command_queue: cl_command_queue) -> CLResult<()> {
    // clReleaseCommandQueue performs an implicit flush to issue any previously queued OpenCL
    // commands in command_queue.
    Queue::ref_from_raw(command_queue)?.flush(false)?;
    Queue::release(command_queue)
}
