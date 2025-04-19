#![feature(portable_simd)]

mod lin;

use lin::LinearSpace;
use rayon::prelude::*;
use std::{
  collections::{HashMap, HashSet},
  f64::consts::PI,
  fmt::Write,
  ops::{Add, Div, Rem, Sub},
  simd::{f64x2, num::SimdFloat},
};

fn wrap_mod<T>(x: T, modulus: T) -> T
where
  T: Rem<Output = T> + Add<Output = T> + Copy,
{
  ((x % modulus) + modulus) % modulus
}

fn minimum_rotation<T>(a: T, b: T, modulus: T) -> T
where
  T: Rem<Output = T> + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Copy + From<u32>,
{
  let m2 = modulus / T::from(2);
  wrap_mod(b - a + m2, modulus) - m2
}

struct Circle {
  c: f64x2,
  r: f64,
}

impl Circle {
  fn from_radius(p1: f64x2, p2: f64x2, r: f64) -> Option<(Circle, Circle)> {
    let d = p1.distance(p2);
    if d > 2.0 * r || r <= 0.0 {
      return None;
    }
    let mid = (p1 + p2).scale(0.5);
    let h = (r.powi(2) - (d * 0.5).powi(2)).sqrt();
    let perp = (p2 - p1).orthogonal().scale(h / d);
    Some((Circle { c: mid + perp, r }, Circle { c: mid - perp, r }))
  }

  fn circumscribe(p1: f64x2, p2: f64x2, p3: f64x2) -> Option<Circle> {
    let d = 2.0 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]));
    if d.abs() < 1e-10 {
      return None;
    }

    let c = f64x2::from_array([
      (p1.norm().powi(2) * (p2[1] - p3[1])
        + p2.norm().powi(2) * (p3[1] - p1[1])
        + p3.norm().powi(2) * (p1[1] - p2[1]))
        / d,
      (p1.norm().powi(2) * (p3[0] - p2[0])
        + p2.norm().powi(2) * (p1[0] - p3[0])
        + p3.norm().powi(2) * (p2[0] - p1[0]))
        / d,
    ]);

    Some(Circle {
      c,
      r: c.distance(p1),
    })
  }

  fn intersection(c1: &Circle, c2: &Circle) -> Option<(f64x2, f64x2)> {
    let d = c1.c.distance(c2.c);
    if d > c1.r + c2.r || d < (c1.r - c2.r).abs() || d == 0.0 {
      return None;
    }
    let a = (c1.r.powi(2) - c2.r.powi(2) + d.powi(2)) / (2.0 * d);
    let h = (c1.r.powi(2) - a.powi(2)).sqrt();
    let direction = (c2.c - c1.c).scale(1.0 / d);
    let center = c1.c + direction.scale(a);
    let offset = direction.orthogonal().scale(h);
    Some((center + offset, center - offset))
  }
}

struct Graph {
  order: usize,
  edges: Vec<(usize, usize)>,
}

impl Graph {
  fn cut_size(&self, partition: &[bool]) -> usize {
    self
      .edges
      .iter()
      .filter(|&&(i, j)| partition[i] != partition[j])
      .count()
  }

  fn maximum_cut(&self) -> (Vec<bool>, usize) {
    fn fsize(edges: &[usize], partition: usize) -> usize {
      edges
        .iter()
        .filter(|&e| (e & partition).is_power_of_two())
        .count()
    }

    let n = self.order;
    assert!(n < std::mem::size_of::<usize>() * 8);
    let edges: Vec<usize> = self
      .edges
      .iter()
      .map(|&(i, j)| (1usize << i) | (1usize << j))
      .collect();

    // (n - 1) to ignore trivial symmetry, but resulsts in less pretty diagram somehow
    // (0..1usize << n.saturating_sub(1))

    (0..1usize << n)
      .into_par_iter()
      .map(|s| (s, fsize(&edges, s)))
      .max_by_key(|&(_, cut)| cut)
      .map(|(s, c)| ((0..n).map(|i| (s >> i) & 1 != 0).collect(), c))
      .unwrap_or((vec![false; n], 0))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct Point(usize, usize, usize);

impl Point {
  fn torus_l1_distance(p1: Point, p2: Point, m: usize) -> usize {
    let f = |a: usize, b: usize| minimum_rotation(a as i64, b as i64, m as i64).abs() as usize;
    f(p1.0, p2.0) + f(p1.1, p2.1) + f(p1.2, p2.2)
  }
  fn torus_decode(t: usize, m: usize) -> Self {
    let m2 = m * m;
    Point((t - t / m) % m, (t / m - t / m2) % m, (t / m2) % m)
  }
  fn l1_distance(p1: Point, p2: Point) -> usize {
    let f = |a: usize, b: usize| (a as i64 - b as i64).abs() as usize;
    f(p1.0, p2.0) + f(p1.1, p2.1) + f(p1.2, p2.2)
  }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Edge<T>(T, T);

impl<T: Ord> Edge<T> {
  fn new(a: T, b: T) -> Self {
    if a <= b { Self(a, b) } else { Self(b, a) }
  }
}

// Hamiltonian cycle
struct Hamiltonian(Graph);

impl Hamiltonian {
  fn new(order: usize, mut chords: Vec<(usize, usize)>) -> Self {
    chords.sort_by(|(i0, j0), (i1, j1)| i0.cmp(i1).then(j0.cmp(j1)));

    assert!(
      chords
        .iter()
        .all(|&(i, j)| i < order && j < order && i <= j)
    );
    Hamiltonian(Graph {
      order,
      edges: chords,
    })
  }

  fn complete(n: usize) -> Self {
    let chords = (0..n - 1)
      .flat_map(|i| (i + 2..n - (i == 0) as usize).map(move |j| (i, j)))
      .collect();
    Hamiltonian::new(n, chords)
  }

  fn complete_bipartite(n: usize) -> Self {
    let order = 2 * n;
    let chords: Vec<(usize, usize)> = (0..order)
      .flat_map(|i| {
        (i..order)
          .filter(move |&j| {
            minimum_rotation(i as i64, j as i64, order as i64).abs() as usize >= 3
              && j % 2 == (i + 3) % 2
          })
          .map(move |j| (i, j))
      })
      .collect();
    Hamiltonian::new(order, chords)
  }

  fn torus(d: usize, filter: Option<HashSet<Edge<Point>>>) -> Self {
    let filter = filter.as_ref();
    let d3 = d * d * d;
    let chords = (0..d3 - 1)
      .flat_map(|i| {
        let p_i = Point::torus_decode(i, d);
        (i + 2..d3 - (i == 0) as usize)
          .map(move |j| (j, Point::torus_decode(j, d)))
          .filter(move |&(_, p_j)| Point::torus_l1_distance(p_i, p_j, d) == 1)
          .filter(move |&(_, p_j)| filter.is_none_or(|set| set.contains(&Edge::new(p_i, p_j))))
          .map(move |(j, _)| (i, j))
      })
      .collect();
    Hamiltonian::new(d3, chords)
  }

  fn lattice(cycle: Vec<Point>) -> Self {
    assert!(
      cycle
        .iter()
        .zip(cycle.iter().cycle().skip(1))
        .all(|(a, b)| Point::l1_distance(*a, *b) == 1)
    );
    let chords = (0..cycle.len())
      .flat_map(|i| {
        let a = cycle[i];
        (i + 2..cycle.len() - (i == 0) as usize)
          .map(|j| (j, cycle[j]))
          .filter(move |&(_, b)| Point::l1_distance(a, b) == 1)
          .map(move |(j, _)| (i, j))
      })
      .collect();
    Hamiltonian::new(cycle.len(), chords)
  }

  fn multinomial(code: &str) -> Self {
    let elements: Vec<&str> = code
      .split(|c: char| c == ',' || c.is_whitespace())
      .map(|s| s.trim())
      .filter(|s| !s.is_empty())
      .collect();

    let mut start: HashMap<&str, usize> = HashMap::new();
    let mut chords: Vec<(usize, usize)> = Vec::new();

    for (i, &label) in elements.iter().enumerate() {
      if let Some(&prev) = start.get(label) {
        chords.push((prev, i));
        start.remove(label);
      } else {
        start.insert(label, i);
      }
    }

    assert!(start.is_empty(), "Unpaired labels found in Gauss code");

    Hamiltonian::new(elements.len(), chords)
  }

  fn chords(&self) -> &Vec<(usize, usize)> {
    &self.0.edges
  }

  fn intersections(&self) -> Graph {
    let chords = &self.chords();
    let mut intersections = Vec::new();
    for i in 0..chords.len() {
      for j in i + 1..chords.len() {
        let (a, b) = chords[i];
        let (c, d) = chords[j];
        if (a < c && c < b && b < d) || (c < a && a < d && d < b) {
          intersections.push((i, j));
        }
      }
    }
    Graph {
      order: chords.len(),
      edges: intersections,
    }
  }

  fn minimum_crossings(&self) -> (usize, Vec<bool>, Graph) {
    let intersections = self.intersections();
    let (partition, cut) = intersections.maximum_cut();
    (intersections.edges.len() - cut, partition, intersections)
  }

  fn to_svg(&self) -> String {
    let mut svg =
      String::from(r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="-100 -100 200 200">"#);
    svg.push_str(&self.to_svg_group(None, f64x2::splat(0.0)));
    svg.push_str("</svg>");
    svg
  }

  fn to_svg_group(&self, label: Option<String>, translate: f64x2) -> String {
    let n = self.0.order;
    let r = 25.0;
    let node_r = 4.0;
    let inter_r = node_r / 2.0;
    let mut svg = String::new();
    let (_, partition, intersections) = self.minimum_crossings();

    let points: Vec<(f64x2, f64)> = (0..n)
      .map(|i| {
        let angle = 2.0 * PI * i as f64 / n as f64;
        (f64x2::from_array([r * angle.cos(), r * angle.sin()]), angle)
      })
      .collect();

    write!(
      &mut svg,
      r#"<g transform="translate({:.6}, {:.6})">"#,
      translate[0], translate[1]
    )
    .unwrap();

    write!(
      &mut svg,
      r#"<circle cx="0" cy="0" r="{}" stroke="black" stroke-width="1" fill="none"/>"#,
      r
    )
    .unwrap();

    enum Shape {
      Circle(Circle),
      Line(f64x2, f64x2),
    }
    impl Shape {
      fn circle_line_intersection(
        circle: &Circle,
        line_start: f64x2,
        line_end: f64x2,
      ) -> Option<(f64x2, f64x2)> {
        let p1 = line_start - circle.c;
        let p2 = line_end - circle.c;

        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let dr = (dx * dx + dy * dy).sqrt();
        let det = p1[0] * p2[1] - p2[0] * p1[1];
        let disc = circle.r.powi(2) * dr.powi(2) - det.powi(2);

        if disc < 0.0 {
          return None;
        }

        let disc_sqrt = disc.sqrt();
        let sgn = if dy < 0.0 { -1.0 } else { 1.0 };

        let x1 = (det * dy + sgn * dx * disc_sqrt) / (dr * dr);
        let y1 = (-det * dx + dy.abs() * disc_sqrt) / (dr * dr);
        let x2 = (det * dy - sgn * dx * disc_sqrt) / (dr * dr);
        let y2 = (-det * dx - dy.abs() * disc_sqrt) / (dr * dr);

        let pt1 = f64x2::from_array([x1, y1]) + circle.c;
        let pt2 = f64x2::from_array([x2, y2]) + circle.c;

        let t1 = ((pt1 - line_start).dot(p2 - p1)) / (dr * dr);
        let t2 = ((pt2 - line_start).dot(p2 - p1)) / (dr * dr);

        let valid1 = t1 >= 0.0 && t1 <= 1.0;
        let valid2 = t2 >= 0.0 && t2 <= 1.0;

        if valid1 || valid2 {
          Some((pt1, pt2))
        } else {
          None
        }
      }

      fn line_line_intersection(p1: f64x2, p2: f64x2, p3: f64x2, p4: f64x2) -> Option<f64x2> {
        let denom = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]);
        if denom.abs() < 1e-10 {
          return None;
        }

        let t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / denom;
        let u = -((p1[0] - p2[0]) * (p1[1] - p3[1]) - (p1[1] - p2[1]) * (p1[0] - p3[0])) / denom;

        if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
          Some(p1 + (p2 - p1).scale(t))
        } else {
          None
        }
      }
    }

    let mut shapes: Vec<Shape> = Vec::with_capacity(self.0.edges.len());

    for (&(i, j), &inner) in self.0.edges.iter().zip(partition.iter()) {
      let (p_i, angle_i) = points[i];
      let (p_j, angle_j) = points[j];
      let d = minimum_rotation(angle_i, angle_j, 2.0 * PI);
      let theta = angle_i + 0.5 * d;
      let mut offset = 1.0 - (d.abs() / PI).clamp(0.0, 1.0);
      if !inner {
        offset = -offset.clamp(0.25, 1.0);
      }
      let q = f64x2::from_array([offset * r * theta.cos(), offset * r * theta.sin()]);

      if let Some((c1, c2)) =
        Circle::circumscribe(p_i, p_j, q).and_then(|c| Circle::from_radius(p_i, p_j, c.r))
      {
        write!(
                    &mut svg,
                    r#"<path d="M {:.6} {:.6} A {:.6} {:.6} 0 {} {} {:.6} {:.6}" stroke="black" stroke-width="1" fill="none"/>"#,
                    p_i[0], p_i[1], c1.r, c1.r, !inner as u8, ((d <= 0.0) ^ !inner) as u8, p_j[0], p_j[1]
                ).unwrap();
        shapes.push(Shape::Circle(if d <= 0.0 { c1 } else { c2 }));
      } else {
        write!(
          &mut svg,
          r#"<path d="M {:.6} {:.6} L {:.6} {:.6}" stroke="black" stroke-width="1" fill="none"/>"#,
          p_i[0], p_i[1], p_j[0], p_j[1]
        )
        .unwrap();
        shapes.push(Shape::Line(p_i, p_j));
      }
    }

    intersections
      .edges
      .iter()
      .filter(|&&(i, j)| partition[i] == partition[j])
      .for_each(|&(i, j)| match (&shapes[i], &shapes[j]) {
        (Shape::Circle(c1), Shape::Circle(c2)) => {
          if let Some((p1, p2)) = Circle::intersection(c1, c2) {
            let p = if (p1.norm() > p2.norm()) ^ partition[i] {
              p1
            } else {
              p2
            };
            write!(
              &mut svg,
              r#"<circle cx="{}" cy="{}" r="{}" stroke-width="0" fill="black"/>"#,
              p[0], p[1], inter_r
            )
            .unwrap();
          }
        }
        (Shape::Circle(c), Shape::Line(p1, p2)) | (Shape::Line(p1, p2), Shape::Circle(c)) => {
          if let Some((q1, q2)) = Shape::circle_line_intersection(c, *p1, *p2) {
            let p = if q1.norm() < q2.norm() { q1 } else { q2 };
            write!(
              &mut svg,
              r#"<circle cx="{}" cy="{}" r="{}" stroke-width="0" fill="black"/>"#,
              p[0], p[1], inter_r
            )
            .unwrap();
          }
        }
        (Shape::Line(p1, p2), Shape::Line(p3, p4)) => {
          if let Some(p) = Shape::line_line_intersection(*p1, *p2, *p3, *p4) {
            write!(
              &mut svg,
              r#"<circle cx="{}" cy="{}" r="{}" stroke-width="0" fill="black"/>"#,
              p[0], p[1], inter_r
            )
            .unwrap();
          }
        }
      });

    for &(point, _) in &points {
      write!(
        &mut svg,
        r#"<circle cx="{}" cy="{}" r="{}" stroke="black" stroke-width="1" fill="white"/>"#,
        point[0], point[1], node_r
      )
      .unwrap();
    }

    if let Some(l) = label {
      write!(
        &mut svg,
        r#"<text x="0" y="70" text-anchor="middle" font-size="16" font-family="sans-serif" fill="black">{}</text>"#,
        l
      )
      .unwrap();
    }

    svg.push_str("</g>");
    svg
  }
}

fn plot(map: HashMap<(usize, usize), (String, Hamiltonian)>, padding: f64x2) -> String {
  const DIAGRAM_SIZE: f64x2 = f64x2::splat(200.0);

  let max_coords = map.keys().fold(f64x2::splat(0.0), |max, &(x, y)| {
    max.simd_max(f64x2::from_array([x as f64, y as f64]))
  });

  let size = (max_coords + f64x2::splat(1.0)) * DIAGRAM_SIZE + padding * f64x2::splat(2.0);

  let mut svg = String::new();
  write!(
    &mut svg,
    r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{} {} {} {}">"#,
    -size[0] / 2.0,
    -size[1] / 2.0,
    size[0],
    size[1]
  )
  .unwrap();

  write!(
    &mut svg,
    r#"<rect x="{}" y="{}" width="{}" height="{}" fill="white"/>"#,
    -size[0] / 2.0,
    -size[1] / 2.0,
    size[0],
    size[1]
  )
  .unwrap();

  for ((x, y), (label, hamiltonian)) in map {
    let pos = f64x2::from_array([x as f64, y as f64]);
    let translate = (pos - max_coords / f64x2::splat(2.0)) * DIAGRAM_SIZE;
    svg.push_str(&hamiltonian.to_svg_group(Some(label), translate));
  }

  svg.push_str("</svg>");
  svg
}

#[cfg(test)]
mod tests {
  use super::*;

  fn write(path: impl AsRef<str>, content: impl AsRef<str>) {
    let path = std::path::Path::new(path.as_ref());
    if let Some(parent) = path.parent() {
      std::fs::create_dir_all(parent).expect("Failed to create directories");
    }
    std::fs::write(path, content.as_ref()).expect("Failed to write file");
  }

  fn crossing_number_kn(n: usize) -> usize {
    ((n / 2) * ((n - 1) / 2) * ((n - 2) / 2) * ((n - 3) / 2)) / 4
  }

  #[test]
  fn test_complete_edges() {
    for n in 3..8 {
      let (l, e) = (
        n * (n + 1) / 2 - 2 * n,
        Hamiltonian::complete(n).chords().len(),
      );
      assert_eq!(e, l, "K_{}: Chords ({}) must match ({})", n, e, l);
    }
  }

  #[test]
  fn test_crossing_number_complete_graphs() {
    for (n, expected) in (3..8).map(|i| (i, crossing_number_kn(i))) {
      let kn = Hamiltonian::complete(n);
      let (crossings, _, _) = kn.minimum_crossings();
      assert_eq!(
        crossings, expected,
        "K_{}: Computed crossings ({}) should match expected ({})",
        n, crossings, expected
      );
    }
  }

  #[test]
  fn test_complete_bipartite() {
    let cases = vec![("K_3,3", 3, 6, 3), ("K_4,4", 4, 8, 8), ("K_5,5", 5, 10, 15)];
    for (name, n, vertices, chords) in cases {
      let k = Hamiltonian::complete_bipartite(n);
      assert_eq!(k.0.order, vertices);
      assert_eq!(k.chords().len(), chords);
      write(format!("exports/{}.svg", name), k.to_svg());
    }
  }

  #[test]
  fn test_k5_svg() {
    write("exports/k_5.svg", Hamiltonian::complete(5).to_svg());
  }

  #[test]
  fn test_torus() {
    let t2 = Hamiltonian::torus(2, None);
    assert_eq!(t2.chords().len(), 4);
    write("exports/t_2.svg", t2.to_svg());
  }

  #[test]
  fn test_lattice() {
    let cycle = vec![
      Point(0, 0, 0),
      Point(1, 0, 0),
      Point(1, 1, 0),
      Point(0, 1, 0),
      Point(0, 1, 1),
      Point(1, 1, 1),
      Point(1, 0, 1),
      Point(0, 0, 1),
    ];
    let l = Hamiltonian::lattice(cycle);
    assert_eq!(l.chords().len(), 4);
    write("exports/l_1.svg", l.to_svg());
  }

  const PADDING: f64x2 = f64x2::splat(20.0);

  #[test]
  fn test_embedding() {
    let hopf_link = Hamiltonian::lattice(vec![
      Point(2, 0, 1),
      Point(1, 0, 1),
      Point(0, 0, 1),
      Point(0, 1, 1),
      Point(0, 2, 1),
      Point(1, 2, 1),
      Point(2, 2, 1),
      Point(2, 1, 1),
      Point(1, 1, 1),
      Point(1, 1, 2),
      Point(2, 1, 2),
      Point(3, 1, 2),
      Point(3, 1, 1),
      Point(3, 1, 0),
      Point(2, 1, 0),
      Point(1, 1, 0),
      Point(1, 0, 0),
      Point(2, 0, 0),
    ]);

    let trefoil = Hamiltonian::lattice(vec![
      Point(1, 2, 0),
      Point(1, 2, 1),
      Point(1, 2, 2),
      Point(1, 2, 3),
      Point(2, 2, 3),
      Point(3, 2, 3),
      Point(3, 1, 3),
      Point(3, 1, 2),
      Point(3, 1, 1),
      Point(2, 1, 1),
      Point(1, 1, 1),
      Point(0, 1, 1),
      Point(0, 2, 1),
      Point(0, 3, 1),
      Point(0, 3, 2),
      Point(1, 3, 2),
      Point(2, 3, 2),
      Point(2, 2, 2),
      Point(2, 1, 2),
      Point(2, 0, 2),
      Point(2, 0, 1),
      Point(2, 0, 0),
      Point(1, 0, 0),
      Point(1, 1, 0),
    ]);

    // Knots defined within a polyhedra's surface constrain its genus
    // V - E + F = 2 - G (Euler characteristic)

    // Relation between lattice embedding of knots and crossing number within the former
    assert_eq!(hopf_link.minimum_crossings().0, 0); // = 0 is unexpected
    assert_eq!(trefoil.minimum_crossings().0, 11); // ≥ 1 is exptected

    let map = HashMap::from([
      ((0, 0), ("2×2×2".to_owned(), Hamiltonian::torus(2, None))), // Note: non-toroidal and toroidal 2x2x2 lattices are equivalent
      ((1, 0), ("Hopf link".to_owned(), hopf_link)),
      ((2, 0), ("Trefoil".to_owned(), trefoil)),
    ]);

    write("exports/embedding.svg", &plot(map, PADDING));
  }

  #[test]
  fn test_layout() {
    let k_label = |sub: &str| format!("K<tspan dy=\"4\" font-size=\"12\">{}</tspan>", sub);

    let map = HashMap::from([
      ((0, 0), (k_label("3"), Hamiltonian::complete(3))),
      ((1, 0), (k_label("4"), Hamiltonian::complete(4))),
      ((2, 0), (k_label("5"), Hamiltonian::complete(5))),
      ((3, 0), (k_label("6"), Hamiltonian::complete(6))),
      ((0, 1), (k_label("3,3"), Hamiltonian::complete_bipartite(3))),
      ((1, 1), (k_label("4,4"), Hamiltonian::complete_bipartite(4))),
      ((2, 1), (k_label("5,5"), Hamiltonian::complete_bipartite(5))),
      ((3, 1), (k_label("6,6"), Hamiltonian::complete_bipartite(6))),
    ]);
    write("exports/layout.svg", &plot(map, PADDING));
  }

  #[test]
  fn test_multinomial() {
    let cases = vec![
      ("1,2,3,1,2,3", 6, vec![(0, 3), (1, 4), (2, 5)]),
      ("a b a c b c", 6, vec![(0, 2), (1, 4), (3, 5)]),
      ("1 2 3 4 1 3 2 4", 8, vec![(0, 4), (1, 6), (2, 5), (3, 7)]),
    ];

    for (code, expected_order, expected_chords) in cases {
      let h = Hamiltonian::multinomial(code);
      assert_eq!(h.0.order, expected_order, "Order mismatch for {}", code);
      assert_eq!(h.chords(), &expected_chords, "Chords mismatch for {}", code);
    }

    // Experiment with Gauss code of knots
    let map = HashMap::from([
      (
        (0, 0),
        (
          "Trefoil (2,3)".to_owned(),
          Hamiltonian::multinomial("1,2,3,1,2,3"),
        ),
      ),
      (
        (1, 0),
        (
          "Trefoil (2,5)".to_owned(),
          Hamiltonian::multinomial("1,2,3,4,5,1,2,3,4,5"),
        ),
      ),
      (
        (2, 0),
        (
          "Trefoil (2,7)".to_owned(),
          Hamiltonian::multinomial("1,2,3,4,5,6,7,1,2,3,4,5,6,7"),
        ),
      ),
    ]);
    write("exports/nom.svg", &plot(map, PADDING));
  }
}

fn main() {
  println!("Test");
}
