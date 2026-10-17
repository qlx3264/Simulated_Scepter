//! MT19937，兼容 Python 的整数种子、random() 和 _randbelow()。
//! 原算法的许可证见 ../LICENSE-MT19937。

#[derive(Clone)]
pub struct Random {
    state: [u32; 624],
    index: usize,
}

impl Random {
    pub fn new(seed: i64) -> Self {
        let seed = seed.unsigned_abs();
        let words = [seed as u32, (seed >> 32) as u32];
        let words = &words[..if words[1] == 0 { 1 } else { 2 }];
        let mut result = Self {
            state: [0; 624],
            index: 624,
        };
        result.state[0] = 19_650_218;
        for i in 1..624 {
            let previous = result.state[i - 1];
            result.state[i] = 1_812_433_253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(i as u32);
        }
        let (mut i, mut j) = (1, 0);
        for _ in 0..624 {
            let previous = result.state[i - 1];
            result.state[i] = (result.state[i]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(words[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i == 624 {
                result.state[0] = result.state[623];
                i = 1;
            }
            if j == words.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            let previous = result.state[i - 1];
            result.state[i] = (result.state[i]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i == 624 {
                result.state[0] = result.state[623];
                i = 1;
            }
        }
        result.state[0] = 0x8000_0000;
        result
    }

    fn next(&mut self) -> u32 {
        if self.index == 624 {
            for i in 0..624 {
                let combined =
                    (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (combined >> 1)
                    ^ if combined & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    pub fn random(&mut self) -> f64 {
        let a = self.next() >> 5;
        let b = self.next() >> 6;
        (a as f64 * 67_108_864.0 + b as f64) / 9_007_199_254_740_992.0
    }

    pub fn below(&mut self, bound: u64) -> u64 {
        assert!((1..=64).contains(&bound));
        let bits = 64 - bound.leading_zeros();
        loop {
            let value = (self.next() >> (32 - bits)) as u64;
            if value < bound {
                return value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Random;

    #[test]
    fn python_random_zero_seed() {
        let mut rng = Random::new(0);
        assert_eq!(rng.random(), 0.8444218515250481);
        assert_eq!(rng.random(), 0.7579544029403025);
        assert_eq!(rng.random(), 0.420571580830845);
    }

    #[test]
    fn negative_seeds_use_absolute_value() {
        let mut positive = Random::new(8_000_000_001);
        let mut negative = Random::new(-8_000_000_001);
        for _ in 0..2000 {
            assert_eq!(positive.below(6), negative.below(6));
        }
    }
}
