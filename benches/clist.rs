use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use mini::clist::{CList, ROOT_ADDR};

const DATA: [i32; 1024] = [0; 1024];

fn from_slice(c: &mut Criterion) {
    c.bench_function("from_slice", |b| b.iter(|| CList::from_slice(&DATA)));
}

fn insert_iter_right(c: &mut Criterion) {
    c.bench_function("insert_iter_right", |b| {
        b.iter(|| {
            let mut clist = CList::with_capacity(DATA.len());
            clist.insert_iter_right(ROOT_ADDR, DATA)
        })
    });
}

fn insert_iter_left(c: &mut Criterion) {
    c.bench_function("insert_iter_left", |b| {
        b.iter(|| {
            let mut clist = CList::with_capacity(DATA.len());
            clist.insert_iter_left(ROOT_ADDR, DATA)
        })
    });
}

fn remove_n(c: &mut Criterion) {
    c.bench_function("remove_n", |b| {
        let mut clist = CList::with_capacity(DATA.len());
        let _ = clist.insert_iter_right(ROOT_ADDR, DATA);
        b.iter_batched(
            || clist.clone(),
            |mut clist| clist.remove_n(clist.next(ROOT_ADDR), DATA.len()),
            BatchSize::SmallInput,
        )
    });
}

fn collect(c: &mut Criterion) {
    c.bench_function("collect", |b| {
        let mut clist = CList::with_capacity(DATA.len());
        let _ = clist.insert_iter_right(ROOT_ADDR, DATA);
        b.iter(|| clist.collect())
    });
}

criterion_group!(
    clist,
    from_slice,
    insert_iter_right,
    insert_iter_left,
    remove_n,
    collect,
);

criterion_main!(clist);
