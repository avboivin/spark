
use std::cell::RefCell;
use js_sys::{Array, Float32Array, Object, Reflect, Uint8Array, Uint16Array, Uint32Array};
use spark_lib::decoder::{ChunkReceiver, MultiDecoder, SplatEncoding, SplatFileType, SplatGetter};
use spark_lib::gsplat::GsplatArray as GsplatArrayInner;
use spark_lib::csplat::CsplatArray as CsplatArrayInner;
use spark_lib::tsplat::{TsplatArray, Tsplat, TsplatMut};
use wasm_bindgen::prelude::*;
use half::f16;

use crate::ext_splats::ExtSplatsData;
use crate::{decoder::ChunkDecoder, packed_splats::PackedSplatsData};

mod raycast;
use raycast::{raycast_packed_ellipsoids, raycast_ext_ellipsoids};

mod sort;
use sort::{sort_internal, SortBuffers, sort32_internal, Sort32Buffers};

mod decoder;
mod packed_splats;
mod ext_splats;

mod lod_tree;

#[wasm_bindgen(start)]
pub fn wasm_start() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn simd_enabled() -> bool {
    cfg!(target_feature = "simd128")
}

/// Fast Huffman decode using a prefix lookup table (LUT).
/// Reads bits MSB-first per byte (matching the encoder), accumulates
/// by shift-left into a u64 buffer, and uses the LUT built on the JS side.
/// The LUT maps the lut_bits most-significant bits to (code_len, symbol).
#[wasm_bindgen]
pub fn decode_huffman_fast(bytes: &[u8], lut: &[u16], count: u32) -> Vec<u16> {
    let count = count as usize;
    let lut_bits = lut.len().trailing_zeros() as u32;
    if lut_bits == 0 || lut_bits > 24 { return Vec::new(); }
    let mut out = Vec::with_capacity(count);
    // Accumulate bits MSB-first (shift-left). We read bytes one at a time,
    // processing their bits from MSB (bit 7) to LSB (bit 0), shifting each
    // into bit_buf. After filling to >= lut_bits, we look up the MSBs.
    let mut bit_buf: u64 = 0;
    let mut bits_in_buf: u32 = 0;
    let mut byte_idx = 0;

    while out.len() < count && byte_idx < bytes.len() {
        // Fill buffer to at least lut_bits bits
        while bits_in_buf < lut_bits && byte_idx < bytes.len() {
            let b = bytes[byte_idx] as u64;
            // Process bits 7 down to 0 (MSB-first)
            for shift in (0..8).rev() {
                let bit = (b >> shift) & 1;
                bit_buf = (bit_buf << 1) | bit;
                bits_in_buf += 1;
                if bits_in_buf >= lut_bits { break; }
            }
            byte_idx += 1;
        }
        if bits_in_buf == 0 { break; }

        // Look up MSBs in LUT
        let shift = bits_in_buf.saturating_sub(lut_bits);
        let index = if shift < 64 { ((bit_buf >> shift) as usize) & (lut.len() - 1) } else { 0 };
        let entry = lut[index];
        if entry == 0xFFFF { break; }
        let code_len = (entry >> 8) as u32;
        let symbol = (entry & 0xFF) as u16;
        if code_len == 0 || code_len > bits_in_buf { break; }
        out.push(symbol);
        // Consume code_len bits by masking them out of the buffer
        bits_in_buf -= code_len;
        if bits_in_buf == 0 {
            bit_buf = 0;
        } else {
            let mask = (1u64 << bits_in_buf) - 1;
            bit_buf &= mask;
        }
    }
    out
}

thread_local! {
    static SORT_BUFFERS: RefCell<SortBuffers> = RefCell::new(SortBuffers::default());
    static SORT32_BUFFERS: RefCell<Sort32Buffers> = RefCell::new(Sort32Buffers::default());
}

#[wasm_bindgen]
pub fn sort_splats(
    num_splats: u32, readback: Uint16Array, ordering: Uint32Array,
) -> u32 {
    let max_splats = readback.length() as usize;

    let active_splats = SORT_BUFFERS.with_borrow_mut(|buffers| {
        buffers.ensure_size(max_splats);
        let sub_readback = readback.subarray(0, num_splats);
        sub_readback.copy_to(&mut buffers.readback[..num_splats as usize]);

        let active_splats = match sort_internal(buffers, num_splats as usize) {
            Ok(active_splats) => active_splats,
            Err(err) => {
                wasm_bindgen::throw_str(&format!("{}", err));
            }
        };

        if active_splats > 0 {
            // Copy out ordering result
            let subarray = &buffers.ordering[..active_splats as usize];
            ordering.subarray(0, active_splats).copy_from(&subarray);
        }
        active_splats
    });

    active_splats
}

#[wasm_bindgen]
pub fn sort32_splats(
    num_splats: u32, readback: Uint32Array, ordering: Uint32Array,
) -> u32 {
    let max_splats = readback.length() as usize;

    let active_splats = SORT32_BUFFERS.with_borrow_mut(|buffers| {
        buffers.ensure_size(max_splats);
        let sub_readback = readback.subarray(0, num_splats);
        sub_readback.copy_to(&mut buffers.readback[..num_splats as usize]);

        let active_splats = match sort32_internal(buffers, max_splats, num_splats as usize) {
            Ok(active_splats) => active_splats,
            Err(err) => {
                wasm_bindgen::throw_str(&format!("{}", err));
            }
        };

        if active_splats > 0 {
            // Copy out ordering result
            let subarray = &buffers.ordering[..active_splats as usize];
            ordering.subarray(0, active_splats).copy_from(&subarray);
        }
        active_splats
    });

    active_splats
}

#[wasm_bindgen]
pub fn decode_to_packedsplats(
    file_type: Option<String>, path_name: Option<String>, encoding: JsValue,
    sh1_codes: Option<Uint32Array>, sh2_codes: Option<Uint32Array>, sh3_codes: Option<Uint32Array>,
) -> Result<ChunkDecoder, JsValue> {
    let encoding = if encoding.is_falsy() {
        SplatEncoding::default()
    } else {
        serde_wasm_bindgen::from_value(encoding)?
    };

    let file_type = if let Some(file_type) = file_type {
        match SplatFileType::from_enum_str(&file_type) {
            Ok(file_type) => Some(file_type),
            Err(err) => { return Err(JsValue::from(err.to_string())); },
        }
    } else {
        None
    };

    let mut splats = PackedSplatsData::new(encoding);
    splats.set_sh_codes(sh1_codes, sh2_codes, sh3_codes);

    let decoder = MultiDecoder::new(splats, file_type, path_name.as_deref());
    let on_finish = |receiver: Box<dyn ChunkReceiver>| {
        let decoder: Box<MultiDecoder<PackedSplatsData>> = receiver.into_any().downcast().unwrap();
        let file_type = decoder.file_type.unwrap();
        let object = decoder.into_splats().into_splat_object();
        Reflect::set(&object, &JsValue::from_str("fileType"), &JsValue::from(file_type.to_enum_str())).unwrap();
        Ok(JsValue::from(object))
    };

    let decoder = ChunkDecoder::new(Box::new(decoder), Box::new(on_finish));
    Ok(decoder)
}

#[wasm_bindgen]
pub fn decode_to_extsplats(
    file_type: Option<String>, path_name: Option<String>,
    sh1_codes: Option<Uint32Array>, sh2_codes: Option<Uint32Array>, sh3_codes: Option<Array>,
) -> Result<ChunkDecoder, JsValue> {
    let file_type = if let Some(file_type) = file_type {
        match SplatFileType::from_enum_str(&file_type) {
            Ok(file_type) => Some(file_type),
            Err(err) => { return Err(JsValue::from(err.to_string())); },
        }
    } else {
        None
    };

    let mut splats = ExtSplatsData::new();
    splats.set_sh_codes(sh1_codes, sh2_codes, sh3_codes);

    let decoder = MultiDecoder::new(splats, file_type, path_name.as_deref());
    let on_finish = |receiver: Box<dyn ChunkReceiver>| {
        let decoder: Box<MultiDecoder<ExtSplatsData>> = receiver.into_any().downcast().unwrap();
        let file_type = decoder.file_type.unwrap();
        let object = decoder.into_splats().into_splat_object();
        Reflect::set(&object, &JsValue::from_str("fileType"), &JsValue::from(file_type.to_enum_str())).unwrap();
        Ok(JsValue::from(object))
    };

    let decoder = ChunkDecoder::new(Box::new(decoder), Box::new(on_finish));
    Ok(decoder)
}

#[wasm_bindgen]
#[allow(non_snake_case)]
pub struct GsplatArray {
    pub numSplats: usize,
    pub maxShDegree: usize,
    inner: GsplatArrayInner,
}

impl GsplatArray {
    pub fn new(inner: GsplatArrayInner) -> Self {
        Self {
            numSplats: inner.len(),
            maxShDegree: inner.max_sh_degree,
            inner,
        }
    }
}

#[wasm_bindgen]
impl GsplatArray {
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn center(&mut self) -> Result<Array, JsValue> {
        let n = self.len();
        if n == 0 {
            let res = Array::new();
            res.push(&JsValue::from(0.0));
            res.push(&JsValue::from(0.0));
            res.push(&JsValue::from(0.0));
            return Ok(res);
        }

        // Use the per-axis MEDIAN, not the min/max midpoint. Real-world scenes
        // (especially photogrammetry) routinely contain a handful of extreme
        // floater outliers thousands of units from the actual content. A
        // min/max midpoint is dragged toward those outliers -- e.g. a scene
        // whose bulk sits near the origin but has one outlier at 175000 would
        // get "centered" by ~66821, which then shifts every real splat far out
        // of the f16 range used by the packed splat texture, corrupting the
        // entire scene instead of just the outlier. The median is insensitive
        // to a small number of extreme values.
        let mut xs: Vec<f32> = Vec::with_capacity(n);
        let mut ys: Vec<f32> = Vec::with_capacity(n);
        let mut zs: Vec<f32> = Vec::with_capacity(n);
        for i in 0..n {
            let c = self.inner.get(i).center();
            xs.push(c.x);
            ys.push(c.y);
            zs.push(c.z);
        }
        let median = |v: &mut Vec<f32>| -> f32 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v[v.len() / 2]
        };
        let cx = median(&mut xs);
        let cy = median(&mut ys);
        let cz = median(&mut zs);

        for i in 0..n {
            let mut splat = self.inner.get_mut(i);
            let mut c = splat.center();
            c.x -= cx;
            c.y -= cy;
            c.z -= cz;
            splat.set_center(c);
        }

        let res = Array::new();
        res.push(&JsValue::from(cx));
        res.push(&JsValue::from(cy));
        res.push(&JsValue::from(cz));
        Ok(res)
    }

    pub fn has_lod(&self) -> bool {
        self.inner.has_lod_tree()
    }

    // pub fn quick_lod(&mut self, lod_base: f32, merge_filter: bool) {
    //     spark_lib::quick_lod::compute_lod_tree(&mut self.inner, lod_base, merge_filter, |s| web_sys::console::log_1(&JsValue::from(s)));
    //     // spark_lib::quick_lod::compute_lod_tree(&mut self.inner, lod_base, merge_filter, |_s| {});
    // }

    pub fn tiny_lod(&mut self, lod_base: f32, merge_filter: bool) {
        // let log = |s: &str| web_sys::console::log_1(&JsValue::from(s));
        let log = |_s: &str| {};
        self.inner.remove_invalid();
        spark_lib::tiny_lod::compute_lod_tree(&mut self.inner, lod_base, merge_filter, log);
        self.inner.encode_lod_opacity();
        spark_lib::chunk_tree::chunk_tree(&mut self.inner, 0, log);
    }

    pub fn bhatt_lod(&mut self, lod_base: f32) {
        // let log = |s: &str| web_sys::console::log_1(&JsValue::from(s));
        let log = |_s: &str| {};
        self.inner.remove_invalid();
        spark_lib::bhatt_lod::compute_lod_tree(&mut self.inner, lod_base, log);
        self.inner.encode_lod_opacity();
        spark_lib::chunk_tree::chunk_tree(&mut self.inner, 0, log);
    }

    pub fn to_packedsplats(&self, encoding: JsValue) -> Result<Object, JsValue> {
        let encoding = if encoding.is_falsy() {
            None
        } else {
            Some(serde_wasm_bindgen::from_value(encoding)?)
        };
        let splats = match PackedSplatsData::new_from_tsplat_array(&self.inner, encoding) {
            Err(err) => { return Err(JsValue::from(err.to_string())); },
            Ok(splats) => splats,
        };
        Ok(splats.into_splat_object())
    }

    pub fn to_packedsplats_lod(&self, encoding: JsValue) -> Result<Object, JsValue> {
        let encoding = if encoding.is_falsy() {
            None
        } else {
            Some(serde_wasm_bindgen::from_value(encoding)?)
        };
        let splats = match PackedSplatsData::new_from_tsplat_array_lod(&self.inner, encoding) {
            Err(err) => { return Err(JsValue::from(err.to_string())); },
            Ok(splats) => splats,
        };
        Ok(splats.into_splat_object())
    }

    pub fn to_extsplats(&self) -> Result<Object, JsValue> {
        let splats = match ExtSplatsData::new_from_tsplat_array(&self.inner) {
            Err(err) => { return Err(JsValue::from(err.to_string())); },
            Ok(splats) => splats,
        };
        Ok(splats.into_splat_object())
    }

    pub fn to_extsplats_lod(&self) -> Result<Object, JsValue> {
        let splats = match ExtSplatsData::new_from_tsplat_array_lod(&self.inner) {
            Err(err) => { return Err(JsValue::from(err.to_string())); },
            Ok(splats) => splats,
        };
        Ok(splats.into_splat_object())
    }

    pub fn inject_rgba8(&mut self, rgba: Uint8Array) {
        self.inner.inject_rgba8(&rgba.to_vec());
    }

    pub fn clone_subset(&self, start: usize, count: usize) -> GsplatArray {
        GsplatArray::new(self.inner.clone_subset(start, count))
    }

    pub fn to_spz(&self) -> Result<Uint8Array, JsValue> {
        let num = self.inner.len();
        let mut max_abs = 1.0f32;
        for i in 0..num {
            let c = self.inner.get(i).center();
            let ax = c.x.abs();
            let ay = c.y.abs();
            let az = c.z.abs();
            if ax > max_abs { max_abs = ax; }
            if ay > max_abs { max_abs = ay; }
            if az > max_abs { max_abs = az; }
        }
        let bits: u32 = if max_abs > 1.0 {
            (23u32.saturating_sub(max_abs.log2().ceil() as u32)).min(24)
        } else {
            18
        };
        let encoder = spark_lib::spz::SpzEncoder::new(self.inner.clone())
            .with_fractional_bits(bits as u8);
        let bytes = encoder.encode().map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Uint8Array::from(bytes.as_slice()))
    }

    pub fn extract_attributes(&self) -> ExtractedAttributes {
        let num_points = self.inner.len();
        let mut xyz = Vec::with_capacity(num_points * 3);
        let mut opacity = Vec::with_capacity(num_points);
        let mut rgb = Vec::with_capacity(num_points * 3);
        let mut scales = Vec::with_capacity(num_points * 3);
        let mut quaternions = Vec::with_capacity(num_points * 4);
        let mut sh1 = Vec::with_capacity(num_points * 9);

        for i in 0..num_points {
            let splat = self.inner.get(i);
            let c = splat.center();
            xyz.push(c[0]); xyz.push(c[1]); xyz.push(c[2]);

            opacity.push(splat.opacity());

            let color = splat.rgb();
            rgb.push(color[0]); rgb.push(color[1]); rgb.push(color[2]);

            let s = splat.scales();
            scales.push(s[0]); scales.push(s[1]); scales.push(s[2]);

            let q = splat.quaternion().to_array();
            quaternions.push(q[0]); quaternions.push(q[1]); quaternions.push(q[2]); quaternions.push(q[3]);

            if self.inner.max_sh_degree > 0 && i < self.inner.sh1.len() {
                let sh = self.inner.sh1[i].0;
                for row in 0..3 {
                    for col in 0..3 {
                        sh1.push(sh[row][col].to_f32());
                    }
                }
            }
        }

        ExtractedAttributes {
            num_splats: num_points,
            max_sh: self.inner.max_sh_degree,
            xyz,
            opacity,
            rgb,
            scales,
            quaternions,
            sh1,
        }
    }

    pub fn from_attributes(
        xyz: &[f32],
        opacity: &[f32],
        rgb: &[f32],
        scales: &[f32],
        quaternions: &[f32],
        sh1: Option<Float32Array>,
    ) -> Result<GsplatArray, JsValue> {
        let num_points = opacity.len();
        if xyz.len() != num_points * 3 {
            return Err(JsValue::from_str("Invalid xyz array length"));
        }
        if rgb.len() != num_points * 3 {
            return Err(JsValue::from_str("Invalid rgb array length"));
        }
        if scales.len() != num_points * 3 {
            return Err(JsValue::from_str("Invalid scales array length"));
        }
        if quaternions.len() != num_points * 4 {
            return Err(JsValue::from_str("Invalid quaternions array length"));
        }

        let max_sh_degree = if sh1.is_some() { 1 } else { 0 };
        let mut inner = GsplatArrayInner::new_capacity(num_points, max_sh_degree);

        for i in 0..num_points {
            let center = glam::Vec3A::new(xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]);
            let op = opacity[i];
            let color = glam::Vec3A::new(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
            let s_val = glam::Vec3A::new(scales[i * 3], scales[i * 3 + 1], scales[i * 3 + 2]);
            let q = glam::Quat::from_xyzw(quaternions[i * 4], quaternions[i * 4 + 1], quaternions[i * 4 + 2], quaternions[i * 4 + 3]);
            
            let splat = spark_lib::gsplat::Gsplat::new(center, op, color, s_val, q);
            inner.splats.push(splat);
        }

        if let Some(sh1_arr) = sh1 {
            let sh1_vec = sh1_arr.to_vec();
            if sh1_vec.len() != num_points * 9 {
                return Err(JsValue::from_str("Invalid sh1 array length"));
            }
            for i in 0..num_points {
                let offset = i * 9;
                let mut sh = [[half::f16::ZERO; 3]; 3];
                for row in 0..3 {
                    for col in 0..3 {
                        sh[row][col] = half::f16::from_f32(sh1_vec[offset + row * 3 + col]);
                    }
                }
                inner.sh1.push(spark_lib::gsplat::GsplatSH1(sh));
            }
        }

        Ok(GsplatArray::new(inner))
    }

    pub fn extract_lod_tree(&self) -> Result<Uint32Array, JsValue> {
        let num_points = self.inner.len();
        let out = Uint32Array::new_with_length((num_points * 4) as u32);
        
        let mut buffer = vec![0u32; num_points * 4];
        
        for i in 0..num_points {
            let splat = self.inner.get(i);
            let center = splat.center();
            let opacity = splat.opacity();
            let scales = splat.scales();
            let (child_count, child_start) = self.inner.get_child_count_start(i);
            
            let clamp_center_coord = |val: f32| -> f32 {
                val.clamp(-65504.0, 65504.0)
            };
            
            let center_f16: [f16; 3] = [
                f16::from_f32(clamp_center_coord(center[0])),
                f16::from_f32(clamp_center_coord(center[1])),
                f16::from_f32(clamp_center_coord(center[2])),
            ];
            let avg_scale = (scales[0] + scales[1] + scales[2]) / 3.0;
            let expansion = if opacity <= 1.0 { 1.0 } else {
                let a = opacity * 4.0 - 3.0;
                1.0 + 0.7 * (a - 1.0)
            };
            let size = f16::from_f32(clamp_center_coord(2.0 * expansion * avg_scale));
            
            let i4 = i * 4;
            buffer[i4 + 0] = (center_f16[0].to_bits() as u32) | ((center_f16[1].to_bits() as u32) << 16);
            buffer[i4 + 1] = (center_f16[2].to_bits() as u32) | ((size.to_bits() as u32) << 16);
            buffer[i4 + 2] = (child_count as u32) & 0xffff;
            buffer[i4 + 3] = child_start as u32;
        }
        
        out.copy_from(&buffer);
        Ok(out)
    }
}

#[wasm_bindgen]
pub struct ExtractedAttributes {
    pub num_splats: usize,
    pub max_sh: usize,
    xyz: Vec<f32>,
    opacity: Vec<f32>,
    rgb: Vec<f32>,
    scales: Vec<f32>,
    quaternions: Vec<f32>,
    sh1: Vec<f32>,
}

#[wasm_bindgen]
impl ExtractedAttributes {
    #[wasm_bindgen(getter)]
    pub fn xyz(&self) -> Vec<f32> { self.xyz.clone() }
    #[wasm_bindgen(getter)]
    pub fn opacity(&self) -> Vec<f32> { self.opacity.clone() }
    #[wasm_bindgen(getter)]
    pub fn rgb(&self) -> Vec<f32> { self.rgb.clone() }
    #[wasm_bindgen(getter)]
    pub fn scales(&self) -> Vec<f32> { self.scales.clone() }
    #[wasm_bindgen(getter)]
    pub fn quaternions(&self) -> Vec<f32> { self.quaternions.clone() }
    #[wasm_bindgen(getter)]
    pub fn sh1(&self) -> Vec<f32> { self.sh1.clone() }
}

#[wasm_bindgen]
pub fn decode_to_gsplatarray(file_type: Option<String>, path_name: Option<String>) -> Result<ChunkDecoder, JsValue> {
    let file_type = if let Some(file_type) = file_type {
        match SplatFileType::from_enum_str(&file_type) {
            Ok(file_type) => Some(file_type),
            Err(err) => { return Err(JsValue::from(err.to_string())); },
        }
    } else {
        None
    };

    let splats = GsplatArrayInner::new();
    let decoder = MultiDecoder::new(splats, file_type, path_name.as_deref());
    let on_finish = |receiver: Box<dyn ChunkReceiver>| {
        let decoder: Box<MultiDecoder<GsplatArrayInner>> = receiver.into_any().downcast().unwrap();
        let gsplats = GsplatArray::new(decoder.into_splats());
        Ok(JsValue::from(gsplats))
    };

    let decoder = ChunkDecoder::new(Box::new(decoder), Box::new(on_finish));
    Ok(decoder)
}

#[wasm_bindgen]
pub fn packedsplats_to_gsplatarray(num_splats: u32, packed: Uint32Array, extra: Option<Object>, encoding: JsValue) -> Result<GsplatArray, JsValue> {
    let encoding = serde_wasm_bindgen::from_value(encoding)?;
    let mut receiver = match PackedSplatsData::from_js_arrays(packed, num_splats as usize, extra.as_ref(), encoding) {
        Ok(receiver) => receiver,
        Err(err) => { return Err(JsValue::from(err.to_string())); }
    };
    let splats = match receiver.to_gsplat_array() {
        Ok(inner) => inner,
        Err(err) => { return Err(JsValue::from(err.to_string())); }
    };
    Ok(GsplatArray::new(splats))
}

#[wasm_bindgen]
#[allow(non_snake_case)]
pub struct CsplatArray {
    pub numSplats: usize,
    pub maxShDegree: usize,
    inner: CsplatArrayInner,
}

impl CsplatArray {
    pub fn new(inner: CsplatArrayInner) -> Self {
        Self {
            numSplats: inner.len(),
            maxShDegree: inner.max_sh_degree,
            inner,
        }
    }
}

#[wasm_bindgen]
impl CsplatArray {
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn has_lod(&self) -> bool {
        self.inner.has_children()
    }

    pub fn tiny_lod(&mut self, lod_base: f32, merge_filter: bool) {
        // let log = |s: &str| web_sys::console::log_1(&JsValue::from(s));
        let log = |_s: &str| {};
        self.inner.remove_invalid();
        spark_lib::tiny_lod::compute_lod_tree(&mut self.inner, lod_base, merge_filter, log);
        self.inner.encode_lod_opacity();
        spark_lib::chunk_tree::chunk_tree(&mut self.inner, 0, log);
    }

    pub fn bhatt_lod(&mut self, lod_base: f32) {
        // let log = |s: &str| web_sys::console::log_1(&JsValue::from(s));
        let log = |_s: &str| {};
        self.inner.remove_invalid();
        spark_lib::bhatt_lod::compute_lod_tree(&mut self.inner, lod_base, log);
        self.inner.encode_lod_opacity();
        spark_lib::chunk_tree::chunk_tree(&mut self.inner, 0, log);
    }

    pub fn to_packedsplats(&self) -> Result<Object, JsValue> {
        let encoding = self.inner.encoding.clone();
        let splats = match PackedSplatsData::new_from_tsplat_array(&self.inner, encoding) {
            Err(err) => { return Err(JsValue::from(err.to_string())); },
            Ok(splats) => splats,
        };
        Ok(splats.into_splat_object())
    }

    pub fn to_packedsplats_lod(&self) -> Result<Object, JsValue> {
        let encoding = self.inner.encoding.clone();
        let splats = match PackedSplatsData::new_from_tsplat_array_lod(&self.inner, encoding) {
            Err(err) => { return Err(JsValue::from(err.to_string())); },
            Ok(splats) => splats,
        };
        Ok(splats.into_splat_object())
    }

    pub fn to_extsplats(&self) -> Result<Object, JsValue> {
        let splats = match ExtSplatsData::new_from_tsplat_array(&self.inner) {
            Err(err) => { return Err(JsValue::from(err.to_string())); },
            Ok(splats) => splats,
        };
        Ok(splats.into_splat_object())
    }

    pub fn to_extsplats_lod(&self) -> Result<Object, JsValue> {
        let splats = match ExtSplatsData::new_from_tsplat_array_lod(&self.inner) {
            Err(err) => { return Err(JsValue::from(err.to_string())); },
            Ok(splats) => splats,
        };
        Ok(splats.into_splat_object())
    }

    pub fn inject_rgba8(&mut self, rgba: Uint8Array) {
        self.inner.inject_rgba8(&rgba.to_vec());
    }

    pub fn clone_subset(&self, start: usize, count: usize) -> CsplatArray {
        CsplatArray::new(self.inner.clone_subset(start, count))
    }

    pub fn to_spz(&self) -> Result<Uint8Array, JsValue> {
        let num = self.inner.len();
        let mut max_abs = 1.0f32;
        for i in 0..num {
            let c = self.inner.get(i).center();
            let ax = c.x.abs();
            let ay = c.y.abs();
            let az = c.z.abs();
            if ax > max_abs { max_abs = ax; }
            if ay > max_abs { max_abs = ay; }
            if az > max_abs { max_abs = az; }
        }
        let bits: u32 = if max_abs > 1.0 {
            (23u32.saturating_sub(max_abs.log2().ceil() as u32)).min(24)
        } else {
            18
        };
        let encoder = spark_lib::spz::SpzEncoder::new(self.inner.clone())
            .with_fractional_bits(bits as u8);
        let bytes = encoder.encode().map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Uint8Array::from(bytes.as_slice()))
    }
}

#[wasm_bindgen]
pub fn decode_to_csplatarray(file_type: Option<String>, path_name: Option<String>, encoding: JsValue) -> Result<ChunkDecoder, JsValue> {
    let file_type = if let Some(file_type) = file_type {
        match SplatFileType::from_enum_str(&file_type) {
            Ok(file_type) => Some(file_type),
            Err(err) => { return Err(JsValue::from(err.to_string())); },
        }
    } else {
        None
    };

    let encoding = if encoding.is_falsy() {
        None
    } else {
        Some(serde_wasm_bindgen::from_value(encoding)?)
    };
    let splats = CsplatArrayInner::new_encoding(encoding);
    let decoder = MultiDecoder::new(splats, file_type, path_name.as_deref());
    let on_finish = |receiver: Box<dyn ChunkReceiver>| {
        let decoder: Box<MultiDecoder<CsplatArrayInner>> = receiver.into_any().downcast().unwrap();
        let gsplats = CsplatArray::new(decoder.into_splats());
        Ok(JsValue::from(gsplats))
    };

    let decoder = ChunkDecoder::new(Box::new(decoder), Box::new(on_finish));
    Ok(decoder)
}

#[wasm_bindgen]
pub fn packedsplats_to_csplatarray(num_splats: u32, packed: Uint32Array, extra: Option<Object>, encoding: JsValue) -> Result<CsplatArray, JsValue> {
    let encoding = serde_wasm_bindgen::from_value(encoding)?;
    let mut receiver = match PackedSplatsData::from_js_arrays(packed, num_splats as usize, extra.as_ref(), encoding) {
        Ok(receiver) => receiver,
        Err(err) => { return Err(JsValue::from(err.to_string())); }
    };
    let splats = match receiver.to_csplat_array() {
        Ok(inner) => inner,
        Err(err) => { return Err(JsValue::from(err.to_string())); }
    };
    Ok(CsplatArray::new(splats))
}

#[wasm_bindgen]
pub fn extsplats_to_gsplatarray(num_splats: u32, ext1: Uint32Array, ext2: Uint32Array, extra: Option<Object>) -> Result<GsplatArray, JsValue> {
    let mut receiver = match ExtSplatsData::from_js_arrays([ext1, ext2], num_splats as usize, extra.as_ref()) {
        Ok(receiver) => receiver,
        Err(err) => { return Err(JsValue::from(err.to_string())); }
    };
    let splats = match receiver.to_gsplat_array() {
        Ok(inner) => inner,
        Err(err) => { return Err(JsValue::from(err.to_string())); }
    };
    Ok(GsplatArray::new(splats))
}

#[wasm_bindgen]
pub fn tiny_lod_packedsplats(num_splats: u32, packed: Uint32Array, extra: Option<Object>, lod_base: f32, merge_filter: bool, rgba: Option<Uint8Array>, encoding: JsValue) -> Result<Object, JsValue> {
    let mut gs = packedsplats_to_csplatarray(num_splats, packed, extra, encoding)?;
    if let Some(rgba) = rgba {
        gs.inject_rgba8(rgba);
    }
    gs.tiny_lod(lod_base, merge_filter);
    gs.to_packedsplats_lod()
}

#[wasm_bindgen]
pub fn bhatt_lod_packedsplats(num_splats: u32, packed: Uint32Array, extra: Option<Object>, lod_base: f32, rgba: Option<Uint8Array>, encoding: JsValue) -> Result<Object, JsValue> {
    let mut gs = packedsplats_to_csplatarray(num_splats, packed, extra, encoding)?;
    if let Some(rgba) = rgba {
        gs.inject_rgba8(rgba);
    }
    gs.bhatt_lod(lod_base);
    gs.to_packedsplats_lod()
}

#[wasm_bindgen]
pub fn tiny_lod_extsplats(num_splats: u32, ext1: Uint32Array, ext2: Uint32Array, extra: Option<Object>, lod_base: f32, merge_filter: bool, rgba: Option<Uint8Array>) -> Result<Object, JsValue> {
    let mut gs = extsplats_to_gsplatarray(num_splats, ext1, ext2, extra)?;
    if let Some(rgba) = rgba {
        gs.inject_rgba8(rgba);
    }
    gs.tiny_lod(lod_base, merge_filter);
    gs.to_extsplats_lod()
}

#[wasm_bindgen]
pub fn bhatt_lod_extsplats(num_splats: u32, ext1: Uint32Array, ext2: Uint32Array, extra: Option<Object>, lod_base: f32, rgba: Option<Uint8Array>) -> Result<Object, JsValue> {
    let mut gs = extsplats_to_gsplatarray(num_splats, ext1, ext2, extra)?;
    if let Some(rgba) = rgba {
        gs.inject_rgba8(rgba);
    }
    gs.bhatt_lod(lod_base);
    gs.to_extsplats_lod()
}

const RAYCAST_BUFFER_COUNT: usize = 65536;

thread_local! {
    static RAYCAST_BUFFERS: RefCell<(Vec<u32>, Vec<u32>, Vec<f32>)> = RefCell::new((vec![0; RAYCAST_BUFFER_COUNT * 4], vec![0; RAYCAST_BUFFER_COUNT * 4], vec![0.0; RAYCAST_BUFFER_COUNT]));
}

#[wasm_bindgen]
pub fn get_raycast_buffer() -> Uint32Array {
    RAYCAST_BUFFERS.with_borrow_mut(|(buffer, _, _)| {
        unsafe { Uint32Array::view(&buffer) }
    })
}

#[wasm_bindgen]
pub fn get_raycast_buffer2() -> Uint32Array {
    RAYCAST_BUFFERS.with_borrow_mut(|(_, buffer, _)| {
        unsafe { Uint32Array::view(&buffer) }
    })
}

#[wasm_bindgen]
pub fn raycast_packed_buffer(
    origin_x: f32, origin_y: f32, origin_z: f32,
    dir_x: f32, dir_y: f32, dir_z: f32,
    min_opacity: f32, near: f32, far: f32,
    count: u32,
    ln_scale_min: f32, ln_scale_max: f32, lod_opacity: bool,
) -> Float32Array {
    RAYCAST_BUFFERS.with_borrow_mut(|(buffer, _, distances)| {
        let encoding = SplatEncoding {
            ln_scale_min,
            ln_scale_max,
            lod_opacity,
            ..Default::default()
        };

        distances.clear();
        let subbuffer = &buffer[0..(4 * count as usize)];
        raycast_packed_ellipsoids(
            subbuffer, distances,
            [origin_x, origin_y, origin_z], [dir_x, dir_y, dir_z],
            min_opacity, near, far, &encoding,
        );

        unsafe { Float32Array::view(&distances) }
    })
}

#[wasm_bindgen]
pub fn raycast_ext_buffers(
    origin_x: f32, origin_y: f32, origin_z: f32,
    dir_x: f32, dir_y: f32, dir_z: f32,
    min_opacity: f32, near: f32, far: f32,
    count: u32,
) -> Float32Array {
    RAYCAST_BUFFERS.with_borrow_mut(|(buffer, buffer2, distances)| {
        distances.clear();
        let subbuffer = &buffer[0..(4 * count as usize)];
        let subbuffer2 = &buffer2[0..(4 * count as usize)];
        raycast_ext_ellipsoids(
            subbuffer, subbuffer2, distances,
            [origin_x, origin_y, origin_z], [dir_x, dir_y, dir_z],
            min_opacity, near, far,
        );

        unsafe { Float32Array::view(&distances) }
    })
}

#[wasm_bindgen]
pub fn raycast_packed_splats(
    origin_x: f32, origin_y: f32, origin_z: f32,
    dir_x: f32, dir_y: f32, dir_z: f32,
    min_opacity: f32, near: f32, far: f32,
    num_splats: u32, packed_splats: Uint32Array,
    ln_scale_min: f32, ln_scale_max: f32, lod_opacity: bool,
) -> Float32Array {
    let mut distances = Vec::<f32>::new();
    let encoding = SplatEncoding {
        ln_scale_min,
        ln_scale_max,
        lod_opacity,
        ..Default::default()
    };

    _ = RAYCAST_BUFFERS.with_borrow_mut(|(buffer, _, _)| {
        let mut base = 0;
        while base < num_splats {
            let chunk_size = (RAYCAST_BUFFER_COUNT as u32).min(num_splats - base);
            let subarray = packed_splats.subarray(4 * base, 4 * (base + chunk_size));
            let subbuffer = &mut buffer[0..(4 * chunk_size as usize)];
            subarray.copy_to(subbuffer);

            raycast_packed_ellipsoids(
                subbuffer, &mut distances,
                [origin_x, origin_y, origin_z], [dir_x, dir_y, dir_z],
                min_opacity, near, far, &encoding,
            );

            base += chunk_size;
        }
    });

    let output = Float32Array::new_with_length(distances.len() as u32);
    output.copy_from(&distances);
    output
}

#[wasm_bindgen]
pub fn decode_rad_header(bytes: Uint8Array) -> Result<JsValue, JsValue> {
    let bytes = bytes.to_vec();
    let meta_chunks_start = match spark_lib::rad::decode_rad_header(&bytes) {
        Ok(meta_chunks_start) => meta_chunks_start,
        Err(err) => { return Err(JsValue::from(err.to_string())); }
    };
    if let Some((meta, chunks_start)) = meta_chunks_start {
        let object = js_sys::Object::new();
        Reflect::set(&object, &JsValue::from_str("meta"), &serde_wasm_bindgen::to_value(&meta)?)?;
        Reflect::set(&object, &JsValue::from_str("chunksStart"), &JsValue::from_f64(chunks_start as f64))?;
        Ok(JsValue::from(object))
    } else {
        Ok(JsValue::null())
    }
}

#[wasm_bindgen]
pub fn reconstruct_sp5_chunk(
    xyz_raw: &[f32],
    scale_indices: &[u16],
    rotation_indices: &[u16],
    app_indices: &[u16],
    scale_codebook: &[f32],
    rotation_codebook: &[f32],
    app_codebook: &[f32],
    mlp_cont: &[f32],
    mlp_dc: &[f32],
    mlp_sh: &[f32],
    mlp_opacity: &[f32],
    mlp_offset_w0: &[f32],
    mlp_offset_b0: &[f32],
    mlp_offset_w1: &[f32],
    mlp_offset_b1: &[f32],
    mlp_offset_w2: &[f32],
    mlp_offset_b2: &[f32],
    mlp_offset_w3: &[f32],
    mlp_offset_b3: &[f32],
) -> GsplatArray {
    use spark_lib::sp5::{contract_to_unisphere, get_tcnn_frequency_encoding, run_tcnn_mlp, run_pytorch_mlp, Activation};
    use spark_lib::gsplat::{Gsplat, GsplatSH1, GsplatArray as GsplatArrayInner};
    use half::f16;

    let num_points = xyz_raw.len() / 3;

    let mut gsplats = Vec::with_capacity(num_points);
    let mut sh1_vec = Vec::with_capacity(num_points);

    let sigmoid = |v: f32| -> f32 { 1.0 / (1.0 + (-v).exp()) };

    for i in 0..num_points {
        // NOTE: this loop used to re-derive `idx` by running stable_lexicographic_sort
        // on the decoded xyz_raw and looking up sorted_indices[i], on the theory that
        // the encoder wrote every per-point stream in position-lexicographic order and
        // the decoder needed to recover that permutation from the (quantized) positions
        // alone. That was true for an older chunking scheme; since Phase 1's
        // monolithic-tree-then-slice rework, converter.ts writes position, scale/
        // rotation indices, opacity, dc, and sh for splat `i` into output slot `i` in
        // ALL of these arrays consistently -- there is no encoder-side reordering left
        // to recover, and chunk-internal order is BFS/tree order, not lexicographic.
        // Re-sorting by position was therefore recovering a permutation that never
        // matched the encoder's actual write order, silently pairing each output splat
        // with a different, spatially-nearby splat's scale/rotation/opacity/color.
        // Measured on test/sp5_ordering_test.ts's synthetic worst case: 85.5% of leaf
        // splats got another splat's attributes with the re-sort, vs 10.4% (residual
        // codebook quantization, not mispairing -- see that test's tolerance) using
        // `idx = i` directly. This is what made converted scenes look "completely
        // destroyed" (right positions, wrong size/orientation/color).
        let idx = i;
        let px = xyz_raw[idx * 3 + 0];
        let py = xyz_raw[idx * 3 + 1];
        let pz = xyz_raw[idx * 3 + 2];

        let s_idx0 = scale_indices[idx] as usize;
        let s_idx1 = scale_indices[num_points + idx] as usize;
        let s_idx2 = scale_indices[2 * num_points + idx] as usize;
        let scale_val = [
            scale_codebook[s_idx0],
            scale_codebook[256 + s_idx1],
            scale_codebook[512 + s_idx2],
        ];

        let r_idx0 = rotation_indices[idx] as usize;
        let r_idx1 = rotation_indices[num_points + idx] as usize;
        let rot_val = [
            rotation_codebook[r_idx0 * 2 + 0],
            rotation_codebook[r_idx0 * 2 + 1],
            rotation_codebook[512 + r_idx1 * 2 + 0],
            rotation_codebook[512 + r_idx1 * 2 + 1],
        ];

        let opacity_raw;
        let mut dc_raw;
        let mut sh_raw;
        if mlp_cont.is_empty() {
            opacity_raw = vec![mlp_opacity[idx]];
            dc_raw = vec![mlp_dc[idx * 3 + 0], mlp_dc[idx * 3 + 1], mlp_dc[idx * 3 + 2]];
            sh_raw = vec![0.0f32; 9];
            if !mlp_sh.is_empty() {
                for k in 0..9 {
                    sh_raw[k] = mlp_sh[idx * 9 + k];
                }
            }
        } else {
            let a_idx0 = app_indices[idx] as usize;
            let a_idx1 = app_indices[num_points + idx] as usize;
            let a_idx2 = app_indices[2 * num_points + idx] as usize;
            let app_val = [
                app_codebook[a_idx0 * 2 + 0],
                app_codebook[a_idx0 * 2 + 1],
                app_codebook[512 + a_idx1 * 2 + 0],
                app_codebook[512 + a_idx1 * 2 + 1],
                app_codebook[1024 + a_idx2 * 2 + 0],
                app_codebook[1024 + a_idx2 * 2 + 1],
            ];

            let uni = contract_to_unisphere(px, py, pz);
            let mut encoded_xyz = [0.0f32; 96];
            get_tcnn_frequency_encoding(uni, 16, &mut encoded_xyz);

            let cont_feature = run_tcnn_mlp(&encoded_xyz, mlp_cont, 96, 64, 13, 1, Activation::ReLU);

            let mut space_feature = [0.0f32; 16];
            space_feature[0..13].copy_from_slice(&cont_feature);
            space_feature[13] = app_val[0];
            space_feature[14] = app_val[1];
            space_feature[15] = app_val[2];

            let mut view_feature = [0.0f32; 16];
            view_feature[0..13].copy_from_slice(&cont_feature);
            view_feature[13] = app_val[3];
            view_feature[14] = app_val[4];
            view_feature[15] = app_val[5];

            let opacity_out = run_tcnn_mlp(&space_feature, mlp_opacity, 16, 64, 1, 1, Activation::LeakyReLU);
            opacity_raw = opacity_out;
            dc_raw = run_tcnn_mlp(&space_feature, mlp_dc, 16, 64, 3, 1, Activation::LeakyReLU);
            sh_raw = run_tcnn_mlp(&view_feature, mlp_sh, 16, 64, 9, 1, Activation::LeakyReLU);
        }

        let scale_exp = [scale_val[0].exp(), scale_val[1].exp(), scale_val[2].exp()];
        let scale_mag = (scale_exp[0]*scale_exp[0] + scale_exp[1]*scale_exp[1] + scale_exp[2]*scale_exp[2]).sqrt() + 1e-8;
        let scale_norm = [scale_exp[0] / scale_mag, scale_exp[1] / scale_mag, scale_exp[2] / scale_mag];

        let rot_mag = (rot_val[0]*rot_val[0] + rot_val[1]*rot_val[1] + rot_val[2]*rot_val[2] + rot_val[3]*rot_val[3]).sqrt() + 1e-8;
        let rot_norm = [rot_val[0] / rot_mag, rot_val[1] / rot_mag, rot_val[2] / rot_mag, rot_val[3] / rot_mag];

        let opacity_val;
        if mlp_cont.is_empty() {
            opacity_val = opacity_raw[0];
        } else {
            let mut shs_flat = [0.0f32; 12];
            shs_flat[0..3].copy_from_slice(&dc_raw);
            shs_flat[3..12].copy_from_slice(&sh_raw);
            let mut shs_norm = [0.0f32; 12];
            let shs_mag = shs_flat.iter().map(|&x| x * x).sum::<f32>().sqrt() + 1e-8;
            for j in 0..12 {
                shs_norm[j] = shs_flat[j] / shs_mag;
            }

            let act_opacity = sigmoid(opacity_raw[0]);
            opacity_val = act_opacity;
            let mut shsnn_input = [0.0f32; 23];
            shsnn_input[0..12].copy_from_slice(&shs_norm);
            shsnn_input[12] = act_opacity;
            shsnn_input[13..16].copy_from_slice(&scale_norm);
            shsnn_input[16] = px;
            shsnn_input[17] = py;
            shsnn_input[18] = pz;
            shsnn_input[19..23].copy_from_slice(&rot_norm);

            let feat1 = run_pytorch_mlp(&shsnn_input, mlp_offset_w0, mlp_offset_b0, true);
            let feat2 = run_pytorch_mlp(&feat1, mlp_offset_w1, mlp_offset_b1, true);
            let feat3 = run_pytorch_mlp(&feat2, mlp_offset_w2, mlp_offset_b2, true);
            let sh_offset = run_pytorch_mlp(&feat3, mlp_offset_w3, mlp_offset_b3, false);

            dc_raw[0] += sh_offset[0];
            dc_raw[1] += sh_offset[1];
            dc_raw[2] += sh_offset[2];
            for r in 0..3 {
                sh_raw[r * 3 + 0] += sh_offset[(r + 1) * 3 + 0];
                sh_raw[r * 3 + 1] += sh_offset[(r + 1) * 3 + 1];
                sh_raw[r * 3 + 2] += sh_offset[(r + 1) * 3 + 2];
            }
        }

        // scale_val comes from the SVQ scale codebook, which converter.ts built by
        // quantizing the LINEAR scale values it got from extract_attributes()
        // (ply.rs's PLY parser already applies .exp() to the raw ln-scale PLY
        // properties -- see ply.rs's out_scale assignments -- so extract_attributes()
        // returns linear scale, not log scale). Gsplat.ln_scales, however, is
        // defined to hold ln(scale): every other encode path in this codebase
        // (e.g. gsplat.rs's encode_packed_splat callers) takes .ln() before storing
        // here, and Gsplat::scales() takes .exp() when reading it back. Writing the
        // linear scale_val directly into ln_scales (as before) meant the renderer's
        // later exp(ln_scales) double-exponentiated every splat's size -- e.g. a
        // genuine 5-unit splat became exp(5) ~= 148 units, a 10-unit splat became
        // exp(10) ~= 22026 units -- ballooning the whole scene into one giant,
        // overlapping, oversaturated blob ("a large white ball"). Clamp to a small
        // positive floor before .ln() since codebook values must be positive but a
        // degenerate/zero centroid should not produce -Infinity.
        let scale_val_ln = scale_val.map(|v| v.max(1e-8).ln());

        let gsplat = Gsplat {
            center: glam::Vec3::new(px, py, pz),
            opacity: f16::from_f32(opacity_val),
            rgb: [f16::from_f32(dc_raw[0]), f16::from_f32(dc_raw[1]), f16::from_f32(dc_raw[2])],
            ln_scales: [f16::from_f32(scale_val_ln[0]), f16::from_f32(scale_val_ln[1]), f16::from_f32(scale_val_ln[2])],
            quaternion: [f16::from_f32(rot_val[0]), f16::from_f32(rot_val[1]), f16::from_f32(rot_val[2]), f16::from_f32(rot_val[3])],
        };
        gsplats.push(gsplat);

        let mut sh1 = GsplatSH1::default();
        let mut sh1_arr = [f16::from_f32(0.0); 9];
        for k in 0..9 {
            sh1_arr[k] = f16::from_f32(sh_raw[k]);
        }
        sh1.0 = [
            [sh1_arr[0], sh1_arr[1], sh1_arr[2]],
            [sh1_arr[3], sh1_arr[4], sh1_arr[5]],
            [sh1_arr[6], sh1_arr[7], sh1_arr[8]],
        ];
        sh1_vec.push(sh1);
    }

    crate::GsplatArray::new(GsplatArrayInner {
        max_sh_degree: 1,
        splats: gsplats,
        children: vec![smallvec::smallvec![]; num_points],
        sh1: sh1_vec,
        sh2: Vec::new(),
        sh3: Vec::new(),
    })
}

#[wasm_bindgen]
pub fn stable_lexicographic_sort_wasm(xyz: &[f32]) -> Vec<usize> {
    let num_points = xyz.len() / 3;
    let mut points = Vec::with_capacity(num_points);
    for i in 0..num_points {
        points.push([xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]]);
    }
    spark_lib::sp5::stable_lexicographic_sort(&points)
}
