// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_WAVEFILE_SERVER_WAVE_H_
#define FLATBUFFERS_GENERATED_WAVEFILE_SERVER_WAVE_H_

#include "flatbuffers/flatbuffers.h"

namespace Server {
namespace Wave {

struct WaveHeader;
struct WaveHeaderBuilder;

struct WaveFile;
struct WaveFileBuilder;

struct WaveHeader FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef WaveHeaderBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_CHUNKID = 4,
    VT_CHUNKSIZE = 6,
    VT_FORMAT = 8,
    VT_SUBCHUNK1ID = 10,
    VT_SUBCHUNK1SIZE = 12,
    VT_AUDIOFORMAT = 14,
    VT_NUMCHANNELS = 16,
    VT_SAMPLERATE = 18,
    VT_BYTERATE = 20,
    VT_BLOCKALIGN = 22,
    VT_BITSPERSAMPLE = 24,
    VT_SUBCHUNK2ID = 26,
    VT_SUBCHUNK2SIZE = 28
  };
  uint32_t chunkID() const {
    return GetField<uint32_t>(VT_CHUNKID, 0);
  }
  uint32_t chunkSize() const {
    return GetField<uint32_t>(VT_CHUNKSIZE, 0);
  }
  uint32_t format() const {
    return GetField<uint32_t>(VT_FORMAT, 0);
  }
  uint32_t subchunk1ID() const {
    return GetField<uint32_t>(VT_SUBCHUNK1ID, 0);
  }
  uint32_t subchunk1Size() const {
    return GetField<uint32_t>(VT_SUBCHUNK1SIZE, 0);
  }
  uint16_t audioFormat() const {
    return GetField<uint16_t>(VT_AUDIOFORMAT, 0);
  }
  uint16_t numChannels() const {
    return GetField<uint16_t>(VT_NUMCHANNELS, 0);
  }
  uint32_t sampleRate() const {
    return GetField<uint32_t>(VT_SAMPLERATE, 0);
  }
  uint32_t byteRate() const {
    return GetField<uint32_t>(VT_BYTERATE, 0);
  }
  uint16_t blockAlign() const {
    return GetField<uint16_t>(VT_BLOCKALIGN, 0);
  }
  uint16_t bitsPerSample() const {
    return GetField<uint16_t>(VT_BITSPERSAMPLE, 0);
  }
  uint32_t subchunk2ID() const {
    return GetField<uint32_t>(VT_SUBCHUNK2ID, 0);
  }
  uint32_t subchunk2Size() const {
    return GetField<uint32_t>(VT_SUBCHUNK2SIZE, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint32_t>(verifier, VT_CHUNKID) &&
           VerifyField<uint32_t>(verifier, VT_CHUNKSIZE) &&
           VerifyField<uint32_t>(verifier, VT_FORMAT) &&
           VerifyField<uint32_t>(verifier, VT_SUBCHUNK1ID) &&
           VerifyField<uint32_t>(verifier, VT_SUBCHUNK1SIZE) &&
           VerifyField<uint16_t>(verifier, VT_AUDIOFORMAT) &&
           VerifyField<uint16_t>(verifier, VT_NUMCHANNELS) &&
           VerifyField<uint32_t>(verifier, VT_SAMPLERATE) &&
           VerifyField<uint32_t>(verifier, VT_BYTERATE) &&
           VerifyField<uint16_t>(verifier, VT_BLOCKALIGN) &&
           VerifyField<uint16_t>(verifier, VT_BITSPERSAMPLE) &&
           VerifyField<uint32_t>(verifier, VT_SUBCHUNK2ID) &&
           VerifyField<uint32_t>(verifier, VT_SUBCHUNK2SIZE) &&
           verifier.EndTable();
  }
};

struct WaveHeaderBuilder {
  typedef WaveHeader Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_chunkID(uint32_t chunkID) {
    fbb_.AddElement<uint32_t>(WaveHeader::VT_CHUNKID, chunkID, 0);
  }
  void add_chunkSize(uint32_t chunkSize) {
    fbb_.AddElement<uint32_t>(WaveHeader::VT_CHUNKSIZE, chunkSize, 0);
  }
  void add_format(uint32_t format) {
    fbb_.AddElement<uint32_t>(WaveHeader::VT_FORMAT, format, 0);
  }
  void add_subchunk1ID(uint32_t subchunk1ID) {
    fbb_.AddElement<uint32_t>(WaveHeader::VT_SUBCHUNK1ID, subchunk1ID, 0);
  }
  void add_subchunk1Size(uint32_t subchunk1Size) {
    fbb_.AddElement<uint32_t>(WaveHeader::VT_SUBCHUNK1SIZE, subchunk1Size, 0);
  }
  void add_audioFormat(uint16_t audioFormat) {
    fbb_.AddElement<uint16_t>(WaveHeader::VT_AUDIOFORMAT, audioFormat, 0);
  }
  void add_numChannels(uint16_t numChannels) {
    fbb_.AddElement<uint16_t>(WaveHeader::VT_NUMCHANNELS, numChannels, 0);
  }
  void add_sampleRate(uint32_t sampleRate) {
    fbb_.AddElement<uint32_t>(WaveHeader::VT_SAMPLERATE, sampleRate, 0);
  }
  void add_byteRate(uint32_t byteRate) {
    fbb_.AddElement<uint32_t>(WaveHeader::VT_BYTERATE, byteRate, 0);
  }
  void add_blockAlign(uint16_t blockAlign) {
    fbb_.AddElement<uint16_t>(WaveHeader::VT_BLOCKALIGN, blockAlign, 0);
  }
  void add_bitsPerSample(uint16_t bitsPerSample) {
    fbb_.AddElement<uint16_t>(WaveHeader::VT_BITSPERSAMPLE, bitsPerSample, 0);
  }
  void add_subchunk2ID(uint32_t subchunk2ID) {
    fbb_.AddElement<uint32_t>(WaveHeader::VT_SUBCHUNK2ID, subchunk2ID, 0);
  }
  void add_subchunk2Size(uint32_t subchunk2Size) {
    fbb_.AddElement<uint32_t>(WaveHeader::VT_SUBCHUNK2SIZE, subchunk2Size, 0);
  }
  explicit WaveHeaderBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<WaveHeader> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<WaveHeader>(end);
    return o;
  }
};

inline flatbuffers::Offset<WaveHeader> CreateWaveHeader(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint32_t chunkID = 0,
    uint32_t chunkSize = 0,
    uint32_t format = 0,
    uint32_t subchunk1ID = 0,
    uint32_t subchunk1Size = 0,
    uint16_t audioFormat = 0,
    uint16_t numChannels = 0,
    uint32_t sampleRate = 0,
    uint32_t byteRate = 0,
    uint16_t blockAlign = 0,
    uint16_t bitsPerSample = 0,
    uint32_t subchunk2ID = 0,
    uint32_t subchunk2Size = 0) {
  WaveHeaderBuilder builder_(_fbb);
  builder_.add_subchunk2Size(subchunk2Size);
  builder_.add_subchunk2ID(subchunk2ID);
  builder_.add_byteRate(byteRate);
  builder_.add_sampleRate(sampleRate);
  builder_.add_subchunk1Size(subchunk1Size);
  builder_.add_subchunk1ID(subchunk1ID);
  builder_.add_format(format);
  builder_.add_chunkSize(chunkSize);
  builder_.add_chunkID(chunkID);
  builder_.add_bitsPerSample(bitsPerSample);
  builder_.add_blockAlign(blockAlign);
  builder_.add_numChannels(numChannels);
  builder_.add_audioFormat(audioFormat);
  return builder_.Finish();
}

struct WaveFile FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef WaveFileBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_HEADER = 4,
    VT_RAW_DATA = 6
  };
  const Server::Wave::WaveHeader *header() const {
    return GetPointer<const Server::Wave::WaveHeader *>(VT_HEADER);
  }
  const flatbuffers::Vector<int8_t> *raw_data() const {
    return GetPointer<const flatbuffers::Vector<int8_t> *>(VT_RAW_DATA);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_HEADER) &&
           verifier.VerifyTable(header()) &&
           VerifyOffset(verifier, VT_RAW_DATA) &&
           verifier.VerifyVector(raw_data()) &&
           verifier.EndTable();
  }
};

struct WaveFileBuilder {
  typedef WaveFile Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_header(flatbuffers::Offset<Server::Wave::WaveHeader> header) {
    fbb_.AddOffset(WaveFile::VT_HEADER, header);
  }
  void add_raw_data(flatbuffers::Offset<flatbuffers::Vector<int8_t>> raw_data) {
    fbb_.AddOffset(WaveFile::VT_RAW_DATA, raw_data);
  }
  explicit WaveFileBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<WaveFile> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<WaveFile>(end);
    return o;
  }
};

inline flatbuffers::Offset<WaveFile> CreateWaveFile(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<Server::Wave::WaveHeader> header = 0,
    flatbuffers::Offset<flatbuffers::Vector<int8_t>> raw_data = 0) {
  WaveFileBuilder builder_(_fbb);
  builder_.add_raw_data(raw_data);
  builder_.add_header(header);
  return builder_.Finish();
}

inline flatbuffers::Offset<WaveFile> CreateWaveFileDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<Server::Wave::WaveHeader> header = 0,
    const std::vector<int8_t> *raw_data = nullptr) {
  auto raw_data__ = raw_data ? _fbb.CreateVector<int8_t>(*raw_data) : 0;
  return Server::Wave::CreateWaveFile(
      _fbb,
      header,
      raw_data__);
}

inline const Server::Wave::WaveFile *GetWaveFile(const void *buf) {
  return flatbuffers::GetRoot<Server::Wave::WaveFile>(buf);
}

inline const Server::Wave::WaveFile *GetSizePrefixedWaveFile(const void *buf) {
  return flatbuffers::GetSizePrefixedRoot<Server::Wave::WaveFile>(buf);
}

inline bool VerifyWaveFileBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<Server::Wave::WaveFile>(nullptr);
}

inline bool VerifySizePrefixedWaveFileBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<Server::Wave::WaveFile>(nullptr);
}

inline void FinishWaveFileBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<Server::Wave::WaveFile> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedWaveFileBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<Server::Wave::WaveFile> root) {
  fbb.FinishSizePrefixed(root);
}

}  // namespace Wave
}  // namespace Server

#endif  // FLATBUFFERS_GENERATED_WAVEFILE_SERVER_WAVE_H_
