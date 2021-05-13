use crate::{FvdVec, LIMIT};
use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned, Pointer, Shared};
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

const UNDECIDED: u8 = 0;
const FAILED: u8 = 1;
const PASSED: u8 = 2;

pub const DESCRIPTOR: usize = 0b01;
pub const RESIZED: usize = 0b10;

/// A holder for the value in a `FvdVec`.
///
/// Note that the variant in use should be determined by the tag on the
/// pointer to this, for safety.
///
/// SAFETY: Overwriting this with a value of a different type in-place
/// (including descriptors of different types) is unsafe, even if done
/// atomically.
#[repr(C)] // Ensures that all values are at the beginning of the struct
pub union Value<T> {
    data: ManuallyDrop<Node<T>>,
    push: ManuallyDrop<Descriptor<T>>,
}

pub enum ValueEnum<T> {
    Data(Node<T>),
    Push(Descriptor<T>),
}

// TODO: Make reference count intrusive (remove indirection)
#[derive(Ord, PartialOrd, Eq, PartialEq)]
pub struct Node<T> {
    pub val: Arc<T>,
    _align: Align4,
}

pub enum Descriptor<T> {
    Push(PushDescr<T>),
    Pop(PopDescr<T>),
    PopSub(PopSubDescr<T>),
}

pub struct PushDescr<T> {
    state: AtomicU8,
    // SAFETY: self.value may *not* contain a descriptor (this is here to
    // prevent reallocation when the value is moved into the vec).
    // Once `self.state` is inhabited by `FAILED` or `PASSED`, then this
    // descriptor will not access `self.value`. In particular, this means that
    // `drop()` will never be called on the inhabitant of `value`.
    value: Atomic<Value<T>>,
}

pub struct PopDescr<T> {
    // SAFETY: `child` must follow the tag convention and contain a valid
    // PopSubDescr.
    child: Atomic<Value<T>>,
}

pub struct PopSubDescr<T> {
    // SAFETY: `parent` must follow the tag convention and contain a valid
    // PopDescr.
    parent: Atomic<Value<T>>,
    // SAFETY: `value` must follow the tag convention and contain a valid
    // Value.
    value: Atomic<Value<T>>,
    _align: Align4,
}

#[repr(align(4))]
#[derive(Default, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
struct Align4;

impl<T> Value<T> {
    pub fn new_data(value: T) -> Value<T> {
        Value {
            data: ManuallyDrop::new(Node::new(value)),
        }
    }

    pub fn new_descriptor(value: Descriptor<T>) -> Value<T> {
        Value {
            push: ManuallyDrop::new(value),
        }
    }

    pub fn not_copied(_: &Guard) -> Shared<'_, Value<T>> {
        opaque::not_copied()
    }

    /// Takes the `Value` and adds it to the list of things to be garbage-
    /// collected.
    ///
    /// SAFETY: It must not be possible to access the guard through a later
    /// epoch (same as [`Guard::defer_drop`]. The `Shared` must also follow the
    /// tag convention. It may, however, be null, in which case nothing will
    /// happen.
    pub unsafe fn defer_drop(this: Shared<Value<T>>, guard: &Guard) {
        if !this.is_null() {
            guard.defer_unchecked(move || drop(Value::into_enum(this.into_owned())))
        }
    }

    /// Takes the `Value` and turns it into an enum.
    ///
    /// SAFETY: The `Owned` must follow the tag convention for
    /// `Owned<Value<T>>`.
    #[allow(clippy::wrong_self_convention)]
    pub unsafe fn into_enum(this: Owned<Value<T>>) -> ValueEnum<T> {
        if this.tag() == 0 {
            ValueEnum::Data(ManuallyDrop::into_inner(this.into_box().data))
        } else {
            ValueEnum::Push(ManuallyDrop::into_inner(this.into_box().push))
        }
    }

    /// Takes the `Value` and turns it into its data component.
    ///
    /// SAFETY: This must be known to contain a `Node`, and may not contain a
    /// descriptor.
    pub unsafe fn into_data(self) -> Node<T> {
        ManuallyDrop::into_inner(self.data)
    }

    pub unsafe fn as_descriptor(&self) -> &Descriptor<T> {
        &self.push
    }

    pub unsafe fn as_data(&self) -> &Node<T> {
        &self.data
    }
}

impl<T> Node<T> {
    pub fn new(value: T) -> Node<T> {
        Node {
            val: Arc::new(value),
            _align: Align4,
        }
    }
}

impl<T> Descriptor<T> {
    /// Calls the `complete()` function for the given descriptor.
    ///
    /// SAFETY: To be safe, `this.deref()` must be safe, and `this.deref()`
    /// must point to a valid Descriptor value.
    pub unsafe fn complete_unchecked(
        this: Shared<Value<T>>,
        pos: usize,
        value: &FvdVec<T>,
    ) -> bool {
        assert!(is_descr(this));
        this.deref().push.complete_inner(pos, value, this)
    }

    pub fn as_pop(&self) -> Option<&PopDescr<T>> {
        match self {
            Descriptor::Push(_) => Option::None,
            Descriptor::Pop(x) => Option::Some(x),
            Descriptor::PopSub(_) => Option::None,
        }
    }

    pub fn as_pop_sub(&self) -> Option<&PopSubDescr<T>> {
        match self {
            Descriptor::Push(_) => Option::None,
            Descriptor::Pop(_) => Option::None,
            Descriptor::PopSub(x) => Option::Some(x),
        }
    }

    fn complete_inner(&self, pos: usize, value: &FvdVec<T>, shared_self: Shared<Value<T>>) -> bool {
        assert_eq!(
            self as *const _ as usize,
            shared_self.with_tag(0).into_usize()
        );
        match self {
            Descriptor::Push(push) => push.complete(pos, value, shared_self),
            Descriptor::Pop(pop) => pop.complete(pos, value, shared_self),
            Descriptor::PopSub(pop_sub) => pop_sub.complete(pos, value, shared_self),
        }
    }
}

impl<T> PushDescr<T> {
    /// SAFETY: The tag convention must be followed and `value` must be
    /// inhabited by a `Node`. Furthermore, if `self.complete()` returns
    /// `true`, then this *must* have exclusive access to `value`. If it
    /// returns `false`, then access is relinquished and another thread
    /// may access it, e.g. in order to call `drop()`.
    pub unsafe fn new_value(value: Atomic<Value<T>>) -> PushDescr<T> {
        PushDescr {
            state: AtomicU8::new(UNDECIDED),
            value,
        }
    }

    pub fn complete(&self, pos: usize, value: &FvdVec<T>, shared_self: Shared<Value<T>>) -> bool {
        let guard = &epoch::pin();
        let spot = value.get_spot(pos, guard);
        let spot_2 = value.get_spot(pos - 1, guard);
        let mut failures = 0;
        let mut current = spot_2.load(Ordering::SeqCst, guard);
        while self.state.load(Ordering::SeqCst) == UNDECIDED && is_descr(current) {
            failures += 1;
            if failures > LIMIT {
                let _ = self.state.compare_exchange(
                    UNDECIDED,
                    FAILED,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                );
            }
            // SAFETY: We know current is a descriptor as per the `is_descr()`
            // call earlier.
            // This is safe to dereference because if is_descr() succeeds, then
            // the Shared points to a valid descriptor.
            // NOTE: We don't deal with the descriptor at all because it's the
            // job of the thread that placed the descriptor to clean up after
            // it.
            unsafe { Descriptor::complete_unchecked(current, pos - 1, value) };
            current = spot_2.load(Ordering::SeqCst, guard);
        }
        if self.state.load(Ordering::SeqCst) == UNDECIDED {
            let _ = self.state.compare_exchange(
                UNDECIDED,
                if !current.is_null() { PASSED } else { FAILED },
                Ordering::SeqCst,
                Ordering::SeqCst,
            );
        }
        if self.state.load(Ordering::SeqCst) == PASSED {
            // NOTE: While there is no `unsafe` here, it should be noted that
            // once this is passed over, we no longer "own" the value, so we
            // may not do anything with it, including drop it.
            // NOTE 2: While I originally thought this could be a memory leak
            // if `spot` is swapped out  with a different value (e.g. not the
            // value in `self.value`) before this compare_exchange, I think
            // this is not the case, since any thread planning to insert a
            // value and detects a descriptor will call `complete` on it. Note
            // that if this isn't the case, and in intermediate value is
            // swapped in between the store of `self.state` and this
            // compare_exchange, a memory leak will occur.
            if spot
                .compare_exchange(
                    shared_self,
                    self.value.load(Ordering::SeqCst, guard),
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                    guard,
                )
                .is_ok()
            {
                // SAFETY: Once we've removed this from the vector, it can no
                // longer be accessed by anybody new. It also follows the tag
                // convention.
                unsafe { Value::defer_drop(shared_self, guard) }
            }
        } else {
            // Makes sure this is no longer accessible, so it can be cleaned
            // up, so the given thread can take care of it.
            if spot
                .compare_exchange(
                    shared_self,
                    Shared::null(),
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                    guard,
                )
                .is_ok()
            {
                // SAFETY: Once we've removed this from the vector, it can no
                // longer be accessed by anybody new. It also follows the tag
                // convention.
                unsafe { Value::defer_drop(shared_self, guard) }
            }
        }
        self.state.load(Ordering::SeqCst) == PASSED
    }
}

impl<T> PopDescr<T> {
    pub fn new() -> PopDescr<T> {
        PopDescr {
            child: Atomic::null(),
        }
    }

    /// Gets the value contained within self.
    ///
    /// SAFETY: `self` must have been completed successfully, or `self.value`
    /// will be null.
    pub unsafe fn get_value<'a>(&self, guard: &'a Guard) -> &'a Node<T> {
        // SAFETY:
        // * `self.child` is either null, or points to a valid `PopSubDescr`.
        // * `self` was completed successfully, so `self.child` is not null.
        // * `PopSubDescr.value` always points to a valid `Data`, so
        //   dereferencing it is safe.
        // * Even if the guard outlives `self`, it won't be GCed until after
        //   the guard is freed, so there won't be a use-after-free error.
        self.child
            .load(Ordering::SeqCst, guard)
            .deref()
            .as_descriptor()
            .as_pop_sub()
            .unwrap()
            .value
            .load(Ordering::SeqCst, guard)
            .deref()
            .as_data()
    }

    pub fn complete(&self, pos: usize, value: &FvdVec<T>, shared_self: Shared<Value<T>>) -> bool {
        const SUCCESS_TAG: usize = 0;
        const FAILED_TAG: usize = 1;
        let mut failures = 0;
        let guard = &epoch::pin();
        let spot = value.get_spot(pos - 1, guard);
        while self.child.load(Ordering::SeqCst, guard).is_null() {
            failures += 1;
            if failures > LIMIT {
                let failed = Shared::null().with_tag(FAILED_TAG);
                let _ = self.child.compare_exchange(
                    Shared::null(),
                    failed,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                    guard,
                );
            } else {
                let expected = spot.load(Ordering::SeqCst, guard);
                if expected.is_null() {
                    let failed = Shared::null().with_tag(FAILED_TAG);
                    let _ = self.child.compare_exchange(
                        Shared::null(),
                        failed,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                        guard,
                    );
                } else if is_descr(expected) {
                    // SAFETY: `expected` is a known descriptor, so
                    // dereferencing it is safe.
                    unsafe { Descriptor::complete_unchecked(expected, pos - 1, value) };
                } else {
                    // SAFETY: `parent` points to a valid PopDescr (shared_self
                    // is a valid PopDescr), and `value` points to a valid
                    // Value (expected is a known Value)
                    let psh = unsafe {
                        PopSubDescr::new(Atomic::from(shared_self), Atomic::from(expected))
                    };
                    let psh = Value::new_descriptor(Descriptor::PopSub(psh));
                    let psh = Owned::new(psh).into_shared(guard);
                    if let Result::Ok(psh) = spot.compare_exchange(
                        expected,
                        psh,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                        guard,
                    ) {
                        let _ = self.child.compare_exchange(
                            Shared::null(),
                            psh,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                            guard,
                        );
                        let spot_2 = value.get_spot(pos, guard);
                        let _ = spot_2.compare_exchange(
                            shared_self,
                            Shared::null(),
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                            guard,
                        );
                        if self.child.load(Ordering::SeqCst, guard) == psh {
                            let _ = spot.compare_exchange(
                                psh,
                                Shared::null(),
                                Ordering::SeqCst,
                                Ordering::SeqCst,
                                guard,
                            );
                        } else {
                            let _ = spot.compare_exchange(
                                shared_self,
                                expected,
                                Ordering::SeqCst,
                                Ordering::SeqCst,
                                guard,
                            );
                        }
                    }
                }
            }
        }
        self.child.load(Ordering::SeqCst, guard).tag() == SUCCESS_TAG
    }
}

impl<T> PopSubDescr<T> {
    /// Creates a new `PopSubDescr`.
    ///
    /// SAFETY: `parent` must point to a valid `PopDescr<T>`, and `expected`
    /// must point to valid data.
    pub unsafe fn new(parent: Atomic<Value<T>>, value: Atomic<Value<T>>) -> PopSubDescr<T> {
        PopSubDescr {
            parent,
            value,
            _align: Default::default(),
        }
    }

    pub fn complete(&self, pos: usize, value: &FvdVec<T>, shared_self: Shared<Value<T>>) -> bool {
        assert_eq!(self as *const _ as usize, shared_self.into_usize());
        let guard = &epoch::pin();
        let spot = value.get_spot(pos, guard);
        // SAFETY: `self.parent` points to a valid PopDescr, so loading it is
        // safe.
        let ph = unsafe {
            &self
                .parent
                .load(Ordering::SeqCst, guard)
                .deref()
                .as_descriptor()
                .as_pop()
                .unwrap()
                .child
        };
        match ph.compare_exchange(
            Shared::null(),
            shared_self,
            Ordering::SeqCst,
            Ordering::SeqCst,
            guard,
        ) {
            Result::Ok(_) => {
                let _ = spot.compare_exchange(
                    shared_self,
                    Shared::null(),
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                    guard,
                );
                true
            }
            Result::Err(_) => {
                match spot.compare_exchange(
                    shared_self,
                    self.value.load(Ordering::SeqCst, guard),
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                    guard,
                ) {
                    Result::Ok(_) => {}
                    // SAFETY: If this occurs, the value was overwritten before
                    // it could be popped, and it is thus lost. As such, we are
                    // not able to do anything with it, and must drop it in
                    // order to ensure memory is not leaked. Furthermore, it
                    // does follow the tag convention, as we just got it from
                    // the vector.
                    Result::Err(x) => unsafe { Value::defer_drop(x.new, guard) },
                }
                false
            }
        }
    }
}

impl<T> Clone for Node<T> {
    fn clone(&self) -> Self {
        Node {
            val: self.val.clone(),
            _align: self._align.clone(),
        }
    }
}

impl<T> Deref for Node<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.val
    }
}

impl<T> Default for PopDescr<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Decomposes a `Shared` into a `Result` of its components.
///
/// In order to be used safely, `value` must be safe to dereference. For
/// safety of that, see [`Shared::deref`][a].
///
/// [a]: Shared::deref
pub unsafe fn decompose<T>(value: Shared<Value<T>>) -> Result<&Node<T>, &Descriptor<T>> {
    if value.tag() == 0 {
        Result::Ok(&value.deref().data)
    } else {
        Result::Err(&value.deref().push)
    }
}

/// Checks if the given value is a descriptor.
///
/// If this returns `true`, the given value is safe to use as a descriptor, so
/// long as it follows the tag convention.
#[inline]
pub fn is_descr<T>(current: Shared<Value<T>>) -> bool {
    current.tag() & DESCRIPTOR != 0 && !current.is_null()
}

/// Increments the given value atomically if it is not 0.
///
/// This is useful for decrementing reference counts, where if the count is 0,
/// the value is already on the destructor queue.
fn try_increment(num: &AtomicUsize) -> bool {
    num.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
        if x == 0 {
            Option::None
        } else {
            Option::Some(x + 1)
        }
    })
    .is_ok()
}

/// A module for implementing the `NOT_COPIED` sentinel pointer.
///
/// How this works: We want `NOT_COPIED` to have a unique, known address.
/// Implementing it as a constant directly could lead to issues, as the value
/// could alias with another value by accident.
/// To get around this, we use a non-zero-sized static value.
/// While Rust's aliasing rules for statics aren't technically well-defined,
/// there is no way that a static `u8` is going to alias with a non-static
/// `Value<T>` under any circumstances.
mod opaque {
    use crossbeam_epoch::{Pointer, Shared};

    static NOT_COPIED: u8 = 0;

    pub fn not_copied<'a, T>() -> Shared<'a, T> {
        unsafe { Shared::from_usize(&NOT_COPIED as *const u8 as usize) }
    }
}
