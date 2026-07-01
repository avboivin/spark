
use std::cell::RefCell;
use js_sys::{Array, Float32Array, Object, Reflect, Uint8Array, Uint16Array, Uint32Array};
use spark_lib::decoder::{ChunkReceiver, MultiDecoder, SplatEncoding, SplatFileType, SplatGetter};
use spark_lib::gsplat::GsplatArray as GsplatArrayInner;
use spark_lib::csplat::CsplatArray as CsplatArrayInner;
use spark_lib::tsplat::TsplatArray;
use wasm_bindgen::prelude::*;

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
        let encoder = spark_lib::spz::SpzEncoder::new(self.inner.clone());
        let bytes = encoder.encode().map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Uint8Array::from(bytes.as_slice()))
    }
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
        let encoder = spark_lib::spz::SpzEncoder::new(self.inner.clone());
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
    let mut points = Vec::with_capacity(num_points);
    for i in 0..num_points {
        points.push([xyz_raw[i * 3 + 0], xyz_raw[i * 3 + 1], xyz_raw[i * 3 + 2]]);
    }
    let sorted_indices = spark_lib::sp5::stable_lexicographic_sort(&points);

    let mut gsplats = Vec::with_capacity(num_points);
    let mut sh1_vec = Vec::with_capacity(num_points);

    let sigmoid = |v: f32| -> f32 { 1.0 / (1.0 + (-v).exp()) };

    for i in 0..num_points {
        let idx = sorted_indices[i];
        let px = points[idx][0];
        let py = points[idx][1];
        let pz = points[idx][2];

        let uni = contract_to_unisphere(px, py, pz);
        let mut encoded_xyz = [0.0f32; 96];
        get_tcnn_frequency_encoding(uni, 16, &mut encoded_xyz);

        let cont_feature = run_tcnn_mlp(&encoded_xyz, mlp_cont, 96, 64, 13, 1, Activation::ReLU);

        let s_idx0 = scale_indices[i] as usize;
        let s_idx1 = scale_indices[num_points + i] as usize;
        let s_idx2 = scale_indices[2 * num_points + i] as usize;
        let scale_val = [
            scale_codebook[s_idx0],
            scale_codebook[256 + s_idx1],
            scale_codebook[512 + s_idx2],
        ];

        let r_idx0 = rotation_indices[i] as usize;
        let r_idx1 = rotation_indices[num_points + i] as usize;
        let rot_val = [
            rotation_codebook[r_idx0 * 2 + 0],
            rotation_codebook[r_idx0 * 2 + 1],
            rotation_codebook[512 + r_idx1 * 2 + 0],
            rotation_codebook[512 + r_idx1 * 2 + 1],
        ];

        let a_idx0 = app_indices[i] as usize;
        let a_idx1 = app_indices[num_points + i] as usize;
        let a_idx2 = app_indices[2 * num_points + i] as usize;
        let app_val = [
            app_codebook[a_idx0 * 2 + 0],
            app_codebook[a_idx0 * 2 + 1],
            app_codebook[512 + a_idx1 * 2 + 0],
            app_codebook[512 + a_idx1 * 2 + 1],
            app_codebook[1024 + a_idx2 * 2 + 0],
            app_codebook[1024 + a_idx2 * 2 + 1],
        ];

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

        let opacity_raw = run_tcnn_mlp(&space_feature, mlp_opacity, 16, 64, 1, 1, Activation::LeakyReLU);
        let mut dc_raw = run_tcnn_mlp(&space_feature, mlp_dc, 16, 64, 3, 1, Activation::LeakyReLU);
        let mut sh_raw = run_tcnn_mlp(&view_feature, mlp_sh, 16, 64, 9, 1, Activation::LeakyReLU);

        let scale_exp = [scale_val[0].exp(), scale_val[1].exp(), scale_val[2].exp()];
        let scale_mag = (scale_exp[0]*scale_exp[0] + scale_exp[1]*scale_exp[1] + scale_exp[2]*scale_exp[2]).sqrt() + 1e-8;
        let scale_norm = [scale_exp[0] / scale_mag, scale_exp[1] / scale_mag, scale_exp[2] / scale_mag];

        let rot_mag = (rot_val[0]*rot_val[0] + rot_val[1]*rot_val[1] + rot_val[2]*rot_val[2] + rot_val[3]*rot_val[3]).sqrt() + 1e-8;
        let rot_norm = [rot_val[0] / rot_mag, rot_val[1] / rot_mag, rot_val[2] / rot_mag, rot_val[3] / rot_mag];

        let mut shs_flat = [0.0f32; 12];
        shs_flat[0..3].copy_from_slice(&dc_raw);
        shs_flat[3..12].copy_from_slice(&sh_raw);
        let mut shs_norm = [0.0f32; 12];
        let shs_mag = shs_flat.iter().map(|&x| x * x).sum::<f32>().sqrt() + 1e-8;
        for j in 0..12 {
            shs_norm[j] = shs_flat[j] / shs_mag;
        }

        let act_opacity = sigmoid(opacity_raw[0]);
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

        let gsplat = Gsplat {
            center: glam::Vec3::new(px, py, pz),
            opacity: f16::from_f32(opacity_raw[0]),
            rgb: [f16::from_f32(dc_raw[0]), f16::from_f32(dc_raw[1]), f16::from_f32(dc_raw[2])],
            ln_scales: [f16::from_f32(scale_val[0]), f16::from_f32(scale_val[1]), f16::from_f32(scale_val[2])],
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
