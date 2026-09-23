from __future__ import annotations
from dataclasses import dataclass
import struct
import time
import numpy as np

# Sync, version, channel count, sequence, MCU microseconds, uint16 samples, CRC16-CCITT.
SYNC=b"\xA5\x5A"; HEADER=struct.Struct("<2sBBII")


def crc16_ccitt(data):
    crc=0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8): crc=((crc<<1)^0x1021)&0xFFFF if crc&0x8000 else (crc<<1)&0xFFFF
    return crc


@dataclass(frozen=True)
class EmgPacket:
    sequence:int; mcu_timestamp_us:int; samples:np.ndarray; host_timestamp_s:float


class EmgPacketParser:
    def __init__(self, channels=4): self.channels=channels; self.buffer=bytearray()
    def feed(self,data,host_timestamp_s=None):
        self.buffer.extend(data); packets=[]; size=HEADER.size+2*self.channels+2
        while len(self.buffer)>=size:
            start=self.buffer.find(SYNC)
            if start<0: self.buffer.clear(); break
            if start: del self.buffer[:start]
            if len(self.buffer)<size: break
            frame=bytes(self.buffer[:size]); sync,version,n,seq,stamp=HEADER.unpack_from(frame)
            if version!=1 or n!=self.channels or crc16_ccitt(frame[:-2])!=struct.unpack_from("<H",frame,size-2)[0]: del self.buffer[0]; continue
            samples=np.frombuffer(frame,dtype="<u2",count=n,offset=HEADER.size).copy()
            packets.append(EmgPacket(seq,stamp,samples,time.monotonic() if host_timestamp_s is None else host_timestamp_s)); del self.buffer[:size]
        return packets


class SerialEmgReader:
    def __init__(self,port,baudrate=921600,channels=4): self.port=port; self.baudrate=baudrate; self.parser=EmgPacketParser(channels); self.serial=None
    def connect(self):
        try: import serial
        except ImportError as exc: raise RuntimeError("install pyserial for EMG input") from exc
        self.serial=serial.Serial(self.port,self.baudrate,timeout=.05); return self
    def read(self): return self.parser.feed(self.serial.read(self.serial.in_waiting or 1))
    def close(self):
        if self.serial is not None: self.serial.close()
