#!/usr/bin/env python3
import numpy as np
import json

NEC_ADDRESS_BITS = 16 # 16 bit address or 8 bit address and 8 bit inverted address
NEC_COMMAND_BITS = 16 # Command and inverted command
NEC_BITS = (NEC_ADDRESS_BITS + NEC_COMMAND_BITS)
NEC_UNIT = 560
NEC_HEADER_MARK  = (16 * NEC_UNIT) # 9000
NEC_HEADER_SPACE = (8 * NEC_UNIT)  # 4500
NEC_BIT_MARK     = NEC_UNIT
NEC_ONE_SPACE    = (3 * NEC_UNIT) # 1690
NEC_ZERO_SPACE   = NEC_UNIT

PROTOCOL_IS_LSB_FIRST = False
PROTOCOL_IS_MSB_FIRST = True

def encode_pulse_distance_data(val, nbits, bitmark, onespace, zerospace, msbfirst):
    a = np.array([val],dtype='>i4')
    if msbfirst:
        a = np.unpackbits(a.view('uint8'))
    else:
        a = np.unpackbits(a.view('uint8'), bitorder='little')
    a = a.reshape(-1,1)
    a = np.choose(a, [[NEC_BIT_MARK,NEC_ZERO_SPACE],[NEC_BIT_MARK,NEC_ONE_SPACE]])
    return a.flatten()
        
#code = 0x50af3ec1 # power on
code = 0x50af3fc0 # power off
#code = 0x50AF17E8 # power toggle

encoded = encode_pulse_distance_data(code, NEC_BITS, NEC_BIT_MARK, NEC_ONE_SPACE, NEC_ZERO_SPACE, PROTOCOL_IS_LSB_FIRST)
encoded = np.append([NEC_HEADER_MARK, NEC_HEADER_SPACE], encoded)
encoded = np.append(encoded, [NEC_BIT_MARK, 40000])
data = {
    "format": "us",
    "freq": 38,
    "data": encoded.tolist(),
}
print(json.dumps(data))


