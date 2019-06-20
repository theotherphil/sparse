#![feature(test)]

extern crate test;

use std::collections::HashMap;

fn main() {
    println!("Hello, world!");
}

trait Matrix {
    fn from_dense_array(data: Vec<f64>, rows: usize, cols: usize) -> Self;
    fn multiply(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; x.len()];
        self.multiply_into(x, &mut y);
        y
    }
    fn multiply_into(&self, x: &[f64], y: &mut [f64]);
}

/// Dictionary of keys
#[derive(Debug, Clone)]
struct DOK {
    entries: HashMap<(usize, usize), f64>
}

impl Matrix for DOK {
    fn from_dense_array(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "empty matrices are not supported");

        let mut entries = HashMap::new();

        for row in 0..rows {
            for col in 0..cols {
                let v = data[row * cols + col];
                if v != 0.0 {
                    entries.insert((row, col), v);
                }
            }
        }

        DOK { entries }
    }

    fn multiply_into(&self, x: &[f64], y: &mut [f64]) {
        for ((row, col), value) in &self.entries {
            y[*row] += *value * x[*col];
        }
    }
}

/// List of lists
#[derive(Debug, Clone)]
struct LIL {
    rows: Vec<Vec<(usize, f64)>>
}

impl Matrix for LIL {
    fn from_dense_array(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "empty matrices are not supported");

        let mut rs = Vec::new();

        for row in 0..rows {
            let mut row_entries = Vec::new();
            for col in 0..cols {
                let v = data[row * cols + col];
                if v != 0.0 {
                    row_entries.push((col, v));
                }
            }
            rs.push(row_entries);
        }

        LIL { rows: rs }
    }

    fn multiply_into(&self, x: &[f64], y: &mut [f64]) {
        for row in 0..self.rows.len() {
            for (col, value) in &self.rows[row] {
                y[row] += *value * x[*col];
            }
        }
    }
}

/// Coordinate list
#[derive(Debug, Clone)]
struct COO {
    /// (row, col, value)
    entries: Vec<(usize, usize, f64)>
}

impl Matrix for COO {
    fn from_dense_array(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "empty matrices are not supported");

        let mut entries = Vec::new();

        for row in 0..rows {
            for col in 0..cols {
                let v = data[row * cols + col];
                if v != 0.0 {
                    entries.push((row, col, v));
                }
            }
        }

        COO { entries }
    }

    fn multiply_into(&self, x: &[f64], y: &mut [f64]) {
        for (row, col, value) in &self.entries {
            y[*row] += *value * x[*col];
        }
    }
}

/// Compressed sparse row
#[derive(Debug, Clone)]
struct CSR {
    // All entries in this matrix
    entries: Vec<f64>,
    // row_offsets[i] is the starting index for the ith
    // row. Includes a final entry with value entries.len().
    row_offsets: Vec<usize>,
    // col_indices[i] is the column of the ith member of entries
    col_indices: Vec<usize>
}

impl Matrix for CSR {
    fn from_dense_array(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "empty matrices are not supported");

        let mut entries = Vec::new();
        let mut row_offsets = Vec::new();
        let mut col_indices = Vec::new();

        for row in 0..rows {
            row_offsets.push(entries.len());
            for col in 0..cols {
                let v = data[row * cols + col];
                if v != 0.0 {
                    entries.push(v);
                    col_indices.push(col);
                }
            }
        }
        row_offsets.push(entries.len());

        CSR { entries, row_offsets, col_indices }
    }

    fn multiply_into(&self, x: &[f64], y: &mut [f64]) {
        // Outer loop can be parallelised
        for i in 0..self.row_offsets.len() - 1 {
            for j in self.row_offsets[i]..self.row_offsets[i + 1] {
                y[i] += self.entries[j] * x[self.col_indices[j]];
            }
        }
    }
}

// Row-major dense matrix
struct Dense {
    data: Vec<f64>,
    rows: usize,
    cols: usize
}

impl Matrix for Dense {
    fn from_dense_array(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        Dense { data, rows, cols }
    }

    fn multiply_into(&self, x: &[f64], y: &mut [f64]) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                y[row] += x[col] * self.data[self.cols * row + col];
            }
        }
    }
}

// Also do this one (block based 2005)
// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.70.7389&rep=rep1&type=pdf

// And this one (CSR5 2015)
// https://arxiv.org/pdf/1503.05032.pdf

// And this one (SPARSITY 2003)
// https://pdfs.semanticscholar.org/66ba/6dd1b746aa0e10c3a4f62a5c4dd6b955b503.pdf

#[cfg(test)]
mod tests {
    use super::*;

    struct TestCase {
        x: Vec<f64>,
        m: Vec<f64>,
        r: usize,
        c: usize,
        y: Vec<f64>
    }

    macro_rules! test_multiply {
        ($($name:ident, $matrix_type:ty, $test_case:expr),*) => {
            $(
                #[test]
                fn $name() {
                    let t = $test_case;
                    let d = <$matrix_type>::from_dense_array(t.m, t.r, t.c);
                    let y = d.multiply(&t.x);
                    assert_eq!(y, t.y);
                }
            )*
        }
    }

    macro_rules! bench_multiply {
        ($($name:ident, $matrix_type:ty, $test_case:expr),*) => {
            $(
                #[bench]
                fn $name(b: &mut test::Bencher) {
                    let t = $test_case;
                    let x = t.x;
                    let d = test::black_box(
                        <$matrix_type>::from_dense_array(t.m, t.r, t.c)
                    );
                    let mut y = test::black_box(vec![0.0; x.len()]);
                    b.iter(|| {
                        d.multiply_into(test::black_box(&x), &mut y);
                    });
                }
            )*
        }
    }

    test_multiply!(
        test_dense_t1, Dense, t1(),
        test_lil_t1, LIL, t1(),
        test_coo_t1, COO, t1(),
        test_dok_t1, DOK, t1(),
        test_csr_t1, CSR, t1()
    );

    bench_multiply!(
        bench_dense_t1, Dense, t1(),
        bench_dense_t2, Dense, t2(),
        bench_lil_t1, LIL, t1(),
        bench_lil_t2, LIL, t2(),
        bench_coo_t1, COO, t1(),
        bench_coo_t2, COO, t2(),
        bench_dok_t1, DOK, t1(),
        bench_dok_t2, DOK, t2(),
        bench_csr_t1, CSR, t1(),
        bench_csr_t2, CSR, t2()
    );

    fn t1() -> TestCase {
        TestCase {
            x: vec![1.0, 4.0, 2.0],
            m: vec![
                2.0, -1.0, 7.0,
                0.0, 4.0, 1.0,
                -2.0, 3.0, 5.0
            ],
            r: 3,
            c: 3,
            y: vec![12.0, 18.0, 20.0]
        }
    }

    fn t2() -> TestCase {
        let mut x = Vec::new();
        let mut m = Vec::new();

        let num_rows = 200;
        let num_cols = 2000;

        for row in 0..num_rows {
            for col in 0..num_cols {
                x.push(((row + col) % 15) as f64);
                let v  = if row % 10 == 0 && col % 10 == 0 {
                    (2 * row + col) % 7
                } else {
                    0
                };
                m.push(v as f64);
            }
        }

        TestCase {
            x,
            m,
            r: num_rows,
            c: num_cols,
            y: Vec::new() // TODO
        }
    }

}
