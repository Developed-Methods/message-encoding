use std::borrow::Cow;
use std::io::{Error, ErrorKind, Read, Result, Write};
use std::mem::MaybeUninit;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6};
use std::sync::Arc;
use std::fmt::Debug;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};

pub trait MessageEncoding: Sized {
    const STATIC_SIZE: Option<usize> = None;
    const MAX_SIZE: Option<usize> = Self::STATIC_SIZE;

    const _ASSERT: usize = {
        match (Self::STATIC_SIZE, Self::MAX_SIZE) {
            (Some(a), Some(b)) if a != b => panic!("static size must equal max"),
            (Some(_), None) => panic!("cannot have static and not max"),
            _ => {}
        }
        0
    };

    fn write_to<T: Write>(&self, out: &mut T) -> Result<usize>;

    fn read_from<T: Read>(read: &mut T) -> Result<Self>;

    #[deprecated]
    fn static_size() -> Option<usize> {
        Self::STATIC_SIZE
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct EncodeSkipContext<T, C> {
    pub data: T,
    pub context: C,
}

impl<M: MessageEncoding, C: Default> MessageEncoding for EncodeSkipContext<M, C> {
    const STATIC_SIZE: Option<usize> = M::STATIC_SIZE;

    fn write_to<T: Write>(&self, out: &mut T) -> Result<usize> {
        self.data.write_to(out)
    }

    fn read_from<T: Read>(read: &mut T) -> Result<Self> {
        Ok(EncodeSkipContext {
            data: M::read_from(read)?,
            context: C::default(),
        })
    }
}

pub fn test_assert_valid_encoding<T: MessageEncoding + PartialEq + Debug>(msg: T) {
    assert_eq!(0, T::_ASSERT);

    let mut buffer: Vec<u8> = vec![];
    let bytes_written = msg.write_to(&mut buffer).unwrap();

    assert_eq!(bytes_written, buffer.len());
    if let Some(expected_size) = T::STATIC_SIZE {
        assert_eq!(expected_size, bytes_written);
    }

    if let Some(max_size) = T::MAX_SIZE {
        assert!(bytes_written <= max_size);
    }

    let mut reader = &buffer[..];
    let parsed = T::read_from(&mut reader).unwrap();

    assert_eq!(reader.len(), 0);
    assert_eq!(parsed, msg);
}

impl MessageEncoding for u64 {
    const STATIC_SIZE: Option<usize> = Some(8);

    fn write_to<T: Write>(&self, out: &mut T) -> Result<usize> {
        out.write_u64::<BigEndian>(*self)?;
        Ok(8)
    }

    fn read_from<T: Read>(read: &mut T) -> Result<Self> {
        read.read_u64::<BigEndian>()
    }
}

impl MessageEncoding for u32 {
    const STATIC_SIZE: Option<usize> = Some(4);

    fn write_to<T: Write>(&self, out: &mut T) -> Result<usize> {
        out.write_u32::<BigEndian>(*self)?;
        Ok(4)
    }

    fn read_from<T: Read>(read: &mut T) -> Result<Self> {
        read.read_u32::<BigEndian>()
    }
}

impl MessageEncoding for u16 {
    const STATIC_SIZE: Option<usize> = Some(2);

    fn write_to<T: Write>(&self, out: &mut T) -> Result<usize> {
        out.write_u16::<BigEndian>(*self)?;
        Ok(2)
    }

    fn read_from<T: Read>(read: &mut T) -> Result<Self> {
        read.read_u16::<BigEndian>()
    }
}

impl MessageEncoding for u8 {
    const STATIC_SIZE: Option<usize> = Some(1);

    fn write_to<T: Write>(&self, out: &mut T) -> Result<usize> {
        out.write_u8(*self)?;
        Ok(1)
    }

    fn read_from<T: Read>(read: &mut T) -> Result<Self> {
        read.read_u8()
    }
}

impl<T: MessageEncoding> MessageEncoding for Option<T> {
    const STATIC_SIZE: Option<usize> = match T::STATIC_SIZE {
        Some(v) => Some(v + 1),
        None => None,
    };

    const MAX_SIZE: Option<usize> = match T::MAX_SIZE {
        Some(v) => Some(v + 1),
        None => None,
    };

    fn write_to<I: Write>(&self, out: &mut I) -> Result<usize> {
        match self {
            Some(v) => {
                out.write_u8(1)?;
                Ok(1 + v.write_to(out)?)
            }
            None => {
                out.write_u8(0)?;
                Ok(1)
            }
        }
    }

    fn read_from<I: Read>(read: &mut I) -> Result<Self> {
        match read.read_u8()? {
            0 => Ok(None),
            1 => Ok(Some(T::read_from(read)?)),
            _ => Err(Error::new(ErrorKind::Other, "invalid Option value")),
        }
    }
}

impl<'a, T: MessageEncoding + Clone> MessageEncoding for Cow<'a, T> {
    const STATIC_SIZE: Option<usize> = T::STATIC_SIZE;
    const MAX_SIZE: Option<usize> = T::MAX_SIZE;

    fn write_to<I: Write>(&self, out: &mut I) -> Result<usize> {
        match self {
            Cow::Borrowed(v) => v.write_to(out),
            Cow::Owned(v) => v.write_to(out),
        }
    }

    fn read_from<I: Read>(read: &mut I) -> Result<Self> {
        Ok(Cow::Owned(T::read_from(read)?))
    }
}

impl<T: MessageEncoding> MessageEncoding for Arc<T> {
    const STATIC_SIZE: Option<usize> = T::STATIC_SIZE;
    const MAX_SIZE: Option<usize> = T::MAX_SIZE;

    fn write_to<I: Write>(&self, out: &mut I) -> Result<usize> {
        T::write_to(&*self, out)
    }

    fn read_from<I: Read>(read: &mut I) -> Result<Self> {
        Ok(Arc::new(T::read_from(read)?))
    }
}

impl MessageEncoding for IpAddr {
    const MAX_SIZE: Option<usize> = Some(17);

    fn write_to<T: Write>(&self, out: &mut T) -> Result<usize> {
        match self {
            IpAddr::V4(ip) => {
                out.write_u8(4)?;
                Ok(1 + ip.write_to(out)?)
            }
            IpAddr::V6(ip) => {
                out.write_u8(6)?;
                Ok(1 + ip.write_to(out)?)
            }
        }
    }

    fn read_from<T: Read>(read: &mut T) -> Result<Self> {
        match read.read_u8()? {
            4 => {
                Ok(IpAddr::V4(Ipv4Addr::read_from(read)?))
            }
            6 => {
                Ok(IpAddr::V6(Ipv6Addr::read_from(read)?))
            }
            v => Err(Error::new(ErrorKind::Other, format!("invalid ip type: {}", v))),
        }
    }
}

impl MessageEncoding for SocketAddr {
    const MAX_SIZE: Option<usize> = Some(19);

    fn write_to<T: Write>(&self, out: &mut T) -> Result<usize> {
        match self {
            SocketAddr::V4(addr) => {
                let mut len = 1 + 2;
                out.write_u8(4)?;
                len += addr.ip().write_to(out)?;
                out.write_u16::<BigEndian>(addr.port())?;
                Ok(len)
            }
            SocketAddr::V6(addr) => {
                let mut len = 1 + 2;
                out.write_u8(6)?;
                len += addr.ip().write_to(out)?;
                out.write_u16::<BigEndian>(addr.port())?;
                Ok(len)
            }
        }
    }

    fn read_from<T: Read>(read: &mut T) -> Result<Self> {
        match read.read_u8()? {
            4 => Ok(SocketAddr::V4(SocketAddrV4::new(
                Ipv4Addr::read_from(read)?,
                read.read_u16::<BigEndian>()?,
            ))),
            6 => Ok(SocketAddr::V6(SocketAddrV6::new(
                Ipv6Addr::read_from(read)?,
                read.read_u16::<BigEndian>()?,
                0, 0,
            ))),
            v => Err(Error::new(ErrorKind::Other, format!("invalid ip type: {}", v))),
        }
    }
}

impl MessageEncoding for Ipv4Addr {
    const STATIC_SIZE: Option<usize> = Some(4);

    fn write_to<T: Write>(&self, out: &mut T) -> Result<usize> {
        if out.write(&self.octets())? != 4 {
            return Err(Error::new(ErrorKind::WriteZero, "failed to write full ip"));
        }
        Ok(4)
    }

    fn read_from<T: Read>(read: &mut T) -> Result<Self> {
        let mut bytes = [0u8; 4];
        if read.read(&mut bytes)? != 4 {
            return Err(Error::new(ErrorKind::UnexpectedEof, "missing ip4 data"));
        }
        Ok(Ipv4Addr::from(bytes))
    }
}

impl MessageEncoding for Ipv6Addr {
    const STATIC_SIZE: Option<usize> = Some(16);

    fn write_to<T: Write>(&self, out: &mut T) -> Result<usize> {
        if out.write(&self.octets())? != 16 {
            return Err(Error::new(ErrorKind::WriteZero, "failed to write full ip"));
        }
        Ok(16)
    }

    fn read_from<T: Read>(read: &mut T) -> Result<Self> {
        let mut bytes = [0u8; 16];
        if read.read(&mut bytes)? != 16 {
            return Err(Error::new(ErrorKind::UnexpectedEof, "missing ip6 data"));
        }
        Ok(Ipv6Addr::from(bytes))
    }

    fn static_size() -> Option<usize> {
        Some(16)
    }
}

impl MessageEncoding for SocketAddrV4 {
    const STATIC_SIZE: Option<usize> = Some(m_static::<Ipv4Addr>() + m_static::<u16>());
    
    fn write_to<T: Write>(&self, out: &mut T) -> Result<usize> {
        let mut sum = 0;
        sum += self.ip().write_to(out)?;
        sum += self.port().write_to(out)?;
        Ok(sum)
    }

    fn read_from<T: Read>(read: &mut T) -> Result<Self> {
        Ok(SocketAddrV4::new(Ipv4Addr::read_from(read)?, u16::read_from(read)?))
    }
}

impl MessageEncoding for Vec<u8> {
    fn write_to<T: Write>(&self, out: &mut T) -> Result<usize> {
        out.write_u64::<BigEndian>(self.len() as _)?;
        if out.write(self)? != self.len() {
            return Err(Error::new(ErrorKind::WriteZero, "failed to write entire array"));
        }
        Ok(self.len() + 8)
    }

    fn read_from<T: Read>(read: &mut T) -> Result<Self> {
        let len = read.read_u64::<BigEndian>()? as usize;
        let mut data = vec![0u8; len];
        if read.read(&mut data)? != len {
            return Err(Error::new(ErrorKind::UnexpectedEof, "not enough data for array"));
        }
        Ok(data)
    }
}

impl<T: MessageEncoding, const C: usize> MessageEncoding for [T; C] where [T; C]: Sized {
    const STATIC_SIZE: Option<usize> = match T::STATIC_SIZE {
        Some(v) => Some(C * v),
        None => None,
    };

    const MAX_SIZE: Option<usize> = match T::MAX_SIZE {
        Some(v) => Some(C * v),
        None => None,
    };

    fn write_to<W: Write>(&self, out: &mut W) -> Result<usize> {
        let mut sum = 0;
        for item in self {
            sum += item.write_to(out)?;
        }
        Ok(sum)
    }

    fn read_from<R: Read>(read: &mut R) -> Result<Self> {
        let mut data: [MaybeUninit<T>; C] = unsafe {
            MaybeUninit::uninit().assume_init()
        };

        for elem in &mut data[..] {
            elem.write(T::read_from(read)?);
        }

        Ok(unsafe { array_assume_init(data) })
    }
}

impl<A: MessageEncoding, B: MessageEncoding> MessageEncoding for (A, B) {
    const STATIC_SIZE: Option<usize> = match (A::STATIC_SIZE, B::STATIC_SIZE) {
        (Some(a), Some(b)) => Some(a + b),
        _ => None,
    };

    const MAX_SIZE: Option<usize> = match (A::MAX_SIZE, B::MAX_SIZE) {
        (Some(a), Some(b)) => Some(a + b),
        _ => None,
    };

    fn write_to<W: Write>(&self, out: &mut W) -> Result<usize> {
        let mut sum = 0;
        sum += self.0.write_to(out)?;
        sum += self.1.write_to(out)?;
        Ok(sum)
    }

    fn read_from<R: Read>(read: &mut R) -> Result<Self> {
        Ok((A::read_from(read)?, B::read_from(read)?))
    }
}

impl<'a, T: MessageEncoding> MessageEncoding for &'a T {
    const STATIC_SIZE: Option<usize> = T::STATIC_SIZE;
    const MAX_SIZE: Option<usize> = T::MAX_SIZE;

    fn write_to<W: Write>(&self, out: &mut W) -> Result<usize> {
        T::write_to(self, out)
    }

    fn read_from<R: Read>(_: &mut R) -> Result<Self> {
        Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "cannot read into reference"))
    }
}

// https://github.com/rust-lang/rust/issues/96097
unsafe fn array_assume_init<T, const N: usize>(array: [MaybeUninit<T>; N]) -> [T; N] {
    // SAFETY:
    // * The caller guarantees that all elements of the array are initialized
    // * `MaybeUninit<T>` and T are guaranteed to have the same layout
    // * `MaybeUninit` does not drop, so there are no double-frees
    // And thus the conversion is safe
    let ret = unsafe {
        (&array as *const _ as *const [T; N]).read()
    };

    // FIXME: required to avoid `~const Destruct` bound
    std::mem::forget(array);
    ret
}

impl<'a> MessageEncoding for &'a [u8] {
    fn write_to<T: Write>(&self, out: &mut T) -> std::io::Result<usize> {
        if out.write(self)? != self.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::WriteZero, "not enough space to write raw slice"));
        }
        Ok(self.len())
    }

    fn read_from<T: Read>(_: &mut T) -> std::io::Result<Self> {
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "cannot read for &[u8]"))
    }
}

pub const fn m_static<T: MessageEncoding>() -> usize {
    match T::STATIC_SIZE {
        Some(v) => v,
        None => panic!()
    }
}

pub const fn m_max<T: MessageEncoding>() -> usize {
    match T::MAX_SIZE {
        Some(v) => v,
        None => panic!()
    }
}

pub const fn m_max_list(samples: &'static [usize]) -> usize {
    const fn scan(mut max: usize, idx: usize, samples: &'static [usize]) -> usize {
        if idx == samples.len() {
            return max;
        }

        let compare = samples[idx];
        if max < compare {
            max = compare;
        }

        scan(max, idx + 1, samples)
    }

    if samples.len() == 0 {
        panic!("m_max_list provided 0 samples");
    }
    
    scan(samples[0], 1, samples)
}

#[cfg(test)]
mod test {
    use std::{net::{Ipv4Addr, Ipv6Addr, IpAddr, SocketAddr, SocketAddrV4}, str::FromStr, sync::Arc, borrow::Cow};

    use crate::m_max_list;

    use super::test_assert_valid_encoding;

    #[test]
    fn test_m_max_list() {
        assert_eq!(100, m_max_list(&[3, 5, 67, 1, 51, 100, 54, 1, 65]));
        assert_eq!(67, m_max_list(&[3, 5, 67, 1, 51, 3, 54, 1, 65]));
        assert_eq!(99, m_max_list(&[99, 5, 67, 1, 51, 3, 54, 1, 65]));
        assert_eq!(555, m_max_list(&[99, 5, 67, 1, 51, 3, 54, 1, 555]));
        assert_eq!(99, m_max_list(&[99]));
    }

    #[test]
    fn test_std_encoding() {
        test_assert_valid_encoding(100u64);
        test_assert_valid_encoding(100u32);
        test_assert_valid_encoding(100u16);
        test_assert_valid_encoding(12u8);
        test_assert_valid_encoding(Some(100u16));
        test_assert_valid_encoding(Arc::new(100u16));
        test_assert_valid_encoding(Ipv4Addr::from_str("127.0.0.1").unwrap());
        test_assert_valid_encoding(Ipv6Addr::from_str("203:12::12").unwrap());
        test_assert_valid_encoding(IpAddr::from_str("203:12::12").unwrap());
        test_assert_valid_encoding(IpAddr::from_str("127.0.0.1").unwrap());
        test_assert_valid_encoding(SocketAddr::from_str("127.0.0.1:1234").unwrap());
        test_assert_valid_encoding(SocketAddr::from_str("[203:12::12]:1234").unwrap());
        test_assert_valid_encoding(SocketAddrV4::from_str("127.0.0.1:1234").unwrap());
        test_assert_valid_encoding(Cow::<'_, SocketAddrV4>::Owned(SocketAddrV4::from_str("127.0.0.1:1234").unwrap()));
        test_assert_valid_encoding(vec![1u8, 2, 3, 4]);
        test_assert_valid_encoding([1u8, 2, 3, 4, 5]);

        let v = SocketAddrV4::from_str("127.0.0.1:1234").unwrap();
        test_assert_valid_encoding(Cow::<'_, SocketAddrV4>::Borrowed(&v));
    }
}