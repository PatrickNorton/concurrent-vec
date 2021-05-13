use crate::descr::Value;
use crossbeam_epoch::{Atomic, Pointable, Pointer, Shared};
use std::alloc;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::slice;

pub struct Data<'a, T> {
    previous: &'a Atomic<PartialData<T>>,
    values: &'a [Atomic<Value<T>>],
}

#[repr(transparent)]
pub struct PartialData<T> {
    values: [Atomic<Value<T>>],
}

#[repr(C)]
struct DataInner<T> {
    previous: Atomic<PartialData<T>>,
    len: usize,
    data: [Atomic<Value<T>>; 0],
}

impl<'a, T> Data<'a, T> {
    pub fn get_previous(&self) -> &Atomic<PartialData<T>> {
        &self.previous
    }
}

pub trait DataPointable<T>: Pointable {
    unsafe fn deref_full<'a>(ptr: usize) -> Data<'a, T>;
}

pub trait DataShared<T> {
    unsafe fn deref_full(&self) -> Data<'_, T>;
}

pub trait DataOwned<T> {
    fn deref_full(&self) -> Data<'_, T>;
}

impl<T> Pointable for PartialData<T> {
    const ALIGN: usize = mem::align_of::<DataInner<T>>();
    type Init = usize;

    unsafe fn init(count: Self::Init) -> usize {
        let size = mem::size_of::<DataInner<T>>() + mem::size_of::<Atomic<Value<T>>>() * count;
        let align = mem::align_of::<DataInner<T>>();
        let layout = alloc::Layout::from_size_align(size, align).unwrap();
        let ptr = alloc::alloc_zeroed(layout) as *mut DataInner<T>;
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        (*ptr).len = count;
        ptr as usize
    }

    unsafe fn deref<'a>(ptr: usize) -> &'a Self {
        let array = &*(ptr as *const DataInner<T>);
        &*(slice::from_raw_parts(array.data.as_ptr() as *const _, array.len) as *const _
            as *const PartialData<T>)
    }

    unsafe fn deref_mut<'a>(ptr: usize) -> &'a mut Self {
        let array = &mut *(ptr as *mut DataInner<T>);
        &mut *(slice::from_raw_parts_mut(array.data.as_mut_ptr() as *mut _, array.len) as *mut _
            as *mut PartialData<T>)
    }

    unsafe fn drop(ptr: usize) {
        let array = &*(ptr as *const DataInner<T>);
        let size = mem::size_of::<DataInner<T>>() + mem::size_of::<Atomic<Value<T>>>() * array.len;
        let align = mem::align_of::<DataInner<T>>();
        let layout = alloc::Layout::from_size_align(size, align).unwrap();
        alloc::dealloc(ptr as *mut u8, layout);
    }
}

impl<T> DataPointable<T> for PartialData<T> {
    unsafe fn deref_full<'a>(ptr: usize) -> Data<'a, T> {
        let array = &*(ptr as *const DataInner<T>);
        Data {
            previous: &array.previous,
            values: &<PartialData<_> as Pointable>::deref(ptr).values,
        }
    }
}

impl<'a, T> DataShared<T> for Shared<'a, PartialData<T>> {
    unsafe fn deref_full(&self) -> Data<'a, T> {
        let (ptr, _) = decompose_tag::<T>(self.into_usize());
        PartialData::<T>::deref_full(ptr)
    }
}

impl<T> Deref for PartialData<T> {
    type Target = [Atomic<Value<T>>];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<T> DerefMut for PartialData<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

impl<T> Deref for Data<'_, T> {
    type Target = [Atomic<Value<T>>];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

#[inline]
fn decompose_tag<T: ?Sized + Pointable>(data: usize) -> (usize, usize) {
    (data & !low_bits::<T>(), data & low_bits::<T>())
}

/// Returns a bitmask containing the unused least significant bits of an aligned pointer to `T`.
#[inline]
fn low_bits<T: ?Sized + Pointable>() -> usize {
    (1 << T::ALIGN.trailing_zeros()) - 1
}
