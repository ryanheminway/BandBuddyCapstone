// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_STAGE1_SERVER_STAGE1_H_
#define FLATBUFFERS_GENERATED_STAGE1_SERVER_STAGE1_H_

#include "flatbuffers/flatbuffers.h"

namespace Server {
namespace Stage1 {

struct Stage1;
struct Stage1Builder;

struct Stage1 FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef Stage1Builder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_READY = 4,
    VT_WAVE_DATA_SZ = 6
  };
  uint8_t ready() const {
    return GetField<uint8_t>(VT_READY, 0);
  }
  uint32_t wave_data_sz() const {
    return GetField<uint32_t>(VT_WAVE_DATA_SZ, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint8_t>(verifier, VT_READY) &&
           VerifyField<uint32_t>(verifier, VT_WAVE_DATA_SZ) &&
           verifier.EndTable();
  }
};

struct Stage1Builder {
  typedef Stage1 Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_ready(uint8_t ready) {
    fbb_.AddElement<uint8_t>(Stage1::VT_READY, ready, 0);
  }
  void add_wave_data_sz(uint32_t wave_data_sz) {
    fbb_.AddElement<uint32_t>(Stage1::VT_WAVE_DATA_SZ, wave_data_sz, 0);
  }
  explicit Stage1Builder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<Stage1> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Stage1>(end);
    return o;
  }
};

inline flatbuffers::Offset<Stage1> CreateStage1(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint8_t ready = 0,
    uint32_t wave_data_sz = 0) {
  Stage1Builder builder_(_fbb);
  builder_.add_wave_data_sz(wave_data_sz);
  builder_.add_ready(ready);
  return builder_.Finish();
}

}  // namespace Stage1
}  // namespace Server

#endif  // FLATBUFFERS_GENERATED_STAGE1_SERVER_STAGE1_H_