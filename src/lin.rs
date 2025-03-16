use std::simd::{f64x2, num::SimdFloat};

pub trait LinearSpace<T, const N: usize> {
  fn dot(self, other: Self) -> T;
  fn norm(self) -> T;
  fn distance(self, other: Self) -> T;
  fn angle(self, other: Self) -> T;
  fn scale(self, scalar: T) -> Self;
  fn normalize(self) -> Self;
  fn cross(self, other: Self) -> [T; N];
  fn orthogonal(self) -> Self;
}

impl LinearSpace<f64, 1> for f64x2 {
  fn dot(self, other: Self) -> f64 {
    (self * other).reduce_sum()
  }

  fn norm(self) -> f64 {
    self.dot(self).sqrt()
  }

  fn distance(self, other: Self) -> f64 {
    (self - other).norm()
  }

  fn angle(self, other: Self) -> f64 {
    (self.dot(other) / (self.norm() * other.norm())).acos()
  }

  fn scale(self, scalar: f64) -> Self {
    self * <f64x2>::splat(scalar)
  }

  fn normalize(self) -> Self {
    let n = self.norm();
    if n == 0.0 { self } else { self.scale(1.0 / n) }
  }

  fn cross(self, other: Self) -> [f64; 1] {
    [self[0] * other[1] - self[1] * other[0]]
  }

  fn orthogonal(self) -> Self {
    f64x2::from_array([-self[1], self[0]])
  }
}
