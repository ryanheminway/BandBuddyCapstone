# automatically generated by the FlatBuffers compiler, do not modify

# namespace: Header

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Header(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Header()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsHeader(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Header
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Header
    def PayloadSize(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # Header
    def Destination(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # Header
    def Cmd(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # Header
    def StageId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

def Start(builder): builder.StartObject(4)
def HeaderStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddPayloadSize(builder, payloadSize): builder.PrependUint32Slot(0, payloadSize, 0)
def HeaderAddPayloadSize(builder, payloadSize):
    """This method is deprecated. Please switch to AddPayloadSize."""
    return AddPayloadSize(builder, payloadSize)
def AddDestination(builder, destination): builder.PrependUint8Slot(1, destination, 0)
def HeaderAddDestination(builder, destination):
    """This method is deprecated. Please switch to AddDestination."""
    return AddDestination(builder, destination)
def AddCmd(builder, cmd): builder.PrependUint8Slot(2, cmd, 0)
def HeaderAddCmd(builder, cmd):
    """This method is deprecated. Please switch to AddCmd."""
    return AddCmd(builder, cmd)
def AddStageId(builder, stageId): builder.PrependUint8Slot(3, stageId, 0)
def HeaderAddStageId(builder, stageId):
    """This method is deprecated. Please switch to AddStageId."""
    return AddStageId(builder, stageId)
def End(builder): return builder.EndObject()
def HeaderEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)