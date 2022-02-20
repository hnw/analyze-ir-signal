#!/usr/bin/env python3
import numpy as np
from sklearn.neighbors import KernelDensity

# Relative tolerance (in percent) for some comparisons on measured data.
TOLERANCE = 25
# Lower tolerance for comparison of measured data
LTOL = 100 - TOLERANCE
# Upper tolerance for comparison of measured data
UTOL = 100 + TOLERANCE
# Resolution of the raw input buffer data. Corresponds to 2 pulses of each 26.3 at 38 kHz.
MICROS_PER_TICK = 50
# Value is subtracted from all marks and added to all spaces before decoding, to compensate for the signal forming of different IR receiver modules.
MARK_EXCESS_MICROS = 20

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

def ticks_low(us):
    return us * LTOL / 100

def ticks_high(us):
    return us * UTOL / 100 + 1

def match_mark(us, match):
    passed = ((us >= ticks_low(match + MARK_EXCESS_MICROS)) and (us <= ticks_high(match + MARK_EXCESS_MICROS)));
    return passed

def match_space(us, match):
    passed = ((us >= ticks_low(match - MARK_EXCESS_MICROS)) and (us <= ticks_high(match - MARK_EXCESS_MICROS)));
    return passed

def decode_pulse_distance_data(data, nbits, bitmark, onespace, zerospace, msbfirst):
    a = data[:len(data)-len(data)%2].reshape(-1,2)
    decoded = 0
    if msbfirst:
        for b in a[:nbits]:
            if (not match_mark(b[0], bitmark)):
                print("Mark=%d is not %d" % (b[0], bitmark))
                return False
            if match_space(b[1], onespace):
                decoded = (decoded << 1) | 1
            elif match_space(b[1], zerospace):
                decoded = (decoded << 1) | 0
            else:
                print("Space=%d is not %d or %d" % (b[1], onespace, zerospace))
                return False
    else:
        mask = 1
        i = 0
        for b in a[:nbits]:
            if i % 8 == 0:
                mask = 1
                decoded <<= 8
            if (not match_mark(b[0], bitmark)):
                print("Mark=%d is not %d" % (b[0], bitmark))
                return False
            if match_space(b[1], onespace):
                decoded |= mask
            elif match_space(b[1], zerospace):
                pass
            else:
                print("Space=%d is not %d or %d" % (b[1], onespace, zerospace))
                return False
            mask <<= 1
            i += 1

    print("decoded = %04X" % decoded)
    return decoded

#C8E880=明?

#131780
data_onoff=[3016,1561,344,1186,343,1189,343,425,341,421,348,1185,348,425,341,424,346,419,345,1189,343,1187,342,1188,341,428,342,1184,347,425,344,439,328,423,351,415,351,414,348,428,342,1188,341,436,330,424,348,421,343,422,348,8272,3011,1563,343,1185,344,1183,346,422,346,422,349,1182,346,422,344,425,345,421,349,1185,342,1187,348,1184,342,422,348,1183,346,423,346,419,351,419,348,424,344,427,340,445,323,1190,342,442,325,423,345,422,347,419,348,8272,3014,1559,348,1201,326,1206,326,419,348,425,343,1183,347,419,349,424,343,427,340,1189,343,1186,343,1187,342,422,348,1184,344,436,330,422,351,423,344,424,341,422,348,438,329,1205,324,425,343,422,351,419,348,425,344]

#131720
data_30=[3015,1558,345,1183,346,1201,329,419,347,426,344,1185,347,419,345,444,325,442,328,1201,328,1204,325,1204,330,418,345,1186,343,422,348,422,348,425,342,445,322,1205,325,425,342,426,346,419,345,424,345,440,327,426,345,8262,3012,1562,341,1189,344,1186,341,424,345,424,343,1186,343,429,341,425,343,425,345,1185,344,1186,343,1187,346,425,341,1187,340,440,305,448,354,404,357,419,350,1185,341,426,341,440,329,419,348,426,320,448,345,421,345,8260,3013,1563,343,1187,352,1166,328,450,342,439,328,1208,323,425,343,421,347,422,349,1187,342,1186,352,1165,353,439,330,1204,300,453,341,424,342,428,318,462,333,1203,299,446,346,424,344,428,349,431,301,446,348,424,342]

#131750
data_50=[3020,1555,345,1188,344,1188,342,419,347,424,346,1183,345,419,351,424,342,441,326,1204,328,1201,326,1205,324,422,356,1176,348,419,344,422,350,425,342,1186,343,416,350,1185,345,423,347,419,347,421,348,442,331,418,348,8258,3016,1558,345,1184,346,1183,346,425,342,421,348,1188,343,424,344,417,349,422,348,1185,344,1184,348,1181,346,419,350,1187,342,424,345,424,343,422,350,1183,343,442,328,1201,327,419,349,423,346,424,343,421,349,435,331,8258,3018,1557,346,1184,348,1181,346,422,347,423,344,1186,346,419,354,415,355,407,350,1186,343,1190,343,1183,346,419,348,1203,326,419,350,419,348,424,344,1184,344,422,346,1187,343,417,350,421,348,419,347,442,325,425,346]

#131760
data_100=[3023,1556,346,1183,346,1183,346,419,349,422,348,1184,348,417,352,419,350,420,347,1183,348,1201,328,1185,346,422,345,1185,358,412,345,422,348,419,347,441,328,1187,345,1183,346,425,343,424,346,422,344,440,329,439,332,8267,3017,1558,346,1185,344,1185,344,423,346,423,347,1184,345,420,349,423,346,438,328,1190,342,1184,346,1180,355,416,349,1199,330,419,351,419,351,418,345,422,348,1186,345,1202,327,422,350,414,353,421,348,420,346,419,350,8268,3017,1560,343,1187,345,1202,330,414,352,417,356,1178,348,418,350,417,352,417,350,1184,347,1182,348,1185,344,438,331,1189,343,441,325,417,353,419,362,397,357,1184,346,1184,348,424,342,425,345,419,349,436,331,425,345]

# 070710
data_brighten=[3021,1557,348,1183,348,1183,345,1184,344,423,347,437,330,421,348,421,349,418,348,1184,348,1182,351,1197,329,421,348,421,348,417,350,422,347,423,347,1185,343,419,350,440,327,441,329,421,348,421,353,418,344,423,346,8252,3019,1559,347,1184,343,1184,345,1184,348,417,350,419,350,424,345,439,328,435,335,1201,328,1188,341,1186,347,438,328,441,330,420,347,419,349,423,345,1185,345,427,344,438,328,420,349,416,354,418,348,425,344,424,346,8252,3017,1576,329,1202,327,1186,349,1180,346,419,360,429,326,442,328,422,345,419,350,1184,346,1184,347,1203,330,418,345,422,348,422,351,435,331,437,330,1200,329,419,350,421,345,425,345,419,351,434,332,442,327,421,346,8255,3017,1558,345,1185,348,1199,331,1200,327,418,353,420,347,425,343,422,348,423,344,1189,342,1181,348,1184,345,424,346,436,331,437,332,421,349,420,346,1185,347,425,342,424,346,439,327,419,350,419,348,423,347,422,347,8251,3018,1561,342,1187,342,1185,347,1203,327,417,352,421,348,424,343,419,349,438,349,1155,354,1186,346,1186,344,422,347,419,347,436,333,420,349,424,343,1184,348,421,346,423,346,419,351,438,328,423,346,422,348,421,345]

# 070730
data_darken=[3018,1562,345,1189,372,1147,353,1184,348,423,347,420,346,423,346,424,345,422,347,1185,345,1183,346,1184,348,417,352,419,348,424,346,424,344,423,347,1185,344,1184,345,419,350,424,345,427,343,439,331,414,353,419,349,8261,3018,1558,351,1182,343,1190,350,1176,346,425,345,425,340,422,348,436,333,422,347,1201,328,1186,343,1190,343,424,341,443,327,421,348,417,351,419,352,1181,345,1186,348,415,353,421,345,424,346,419,347,428,341,441,329,8263,3018,1556,347,1188,342,1189,341,1186,346,423,347,440,336,398,361,421,346,428,341,1184,345,1187,342,1185,347,421,349,423,343,423,347,424,344,426,341,1186,343,1189,344,423,343,441,328,417,353,417,349,424,346,423,346]

# 131740
data_nightlight=[3020,1557,348,1185,350,1180,350,419,349,421,348,1185,346,419,348,441,327,442,331,1180,348,1184,348,1188,343,424,346,1186,343,419,350,424,345,424,346,419,350,421,350,1201,329,421,347,417,352,419,350,421,348,421,349,8260,3019,1557,351,1183,346,1184,348,422,347,419,347,1204,327,441,328,441,328,437,335,1184,344,1184,349,1185,346,438,331,1185,348,419,346,426,343,420,350,418,350,420,350,1183,352,418,347,417,354,423,344,422,347,417,354,8259,3019,1557,348,1189,343,1184,346,421,345,440,331,1184,346,422,345,424,345,438,331,1186,346,1186,344,1184,345,419,350,1184,347,421,348,417,353,419,347,424,345,424,346,1184,345,423,346,423,346,427,342,419,353,419,345]

data_hitachi=[8917,4558,525,590,525,1725,526,1728,527,590,524,1727,526,606,513,1726,521,592,527,1725,525,594,528,605,509,1727,526,588,531,1723,527,588,532,1736,511,1726,526,588,531,583,533,593,529,1720,529,1723,526,1730,523,588,530,588,528,1722,528,1727,535,1703,537,586,534,603,512,594,525,1728,522,39873]

def show_aeha(a):
    a = a[:len(a)-len(a)%2].reshape(-1,2)

    trailers = np.where(a>8000)[0] # Trailer >=8ms
    if len(trailers) > 0:
        a = a[:trailers[0]]

    leaders = np.where(a>2800)[0] # T=350-500us, Leader=8T,4T
    if len(leaders) > 0 & leaders[0] == 0:
        print("Trailer = ",a[0])
        a = a[1:]
    
    if len(a) % 8 != 0:
        print("Warning: Data corrupted: bit length = ",len(a))
        return

    if len(a) < 24:
        print("Warning: Data too short: bit length = ",len(a))
        return

    a[a<=500] = 0
    a[(a>500) * (a<=1500)] = 1
    
    customer_code = 0
    for b in a[7::-1]:
        customer_code = customer_code << 1 | b[1]
    for b in a[15:7:-1]:
        customer_code = customer_code << 1 | b[1]

    print("Customer Code = %04X" % customer_code)

    parity = 0
    for b in a[19:15:-1]:
        parity = parity << 1 | b[1]

    print("Parity = 0x%01X" % parity)

    data0 = 0
    for b in a[23:19:-1]:
        data0 = data0 << 1 | b[1]
    
    print("data0 = 0x%01X" % data0)


def decode_nec(rawbuf):
    # Check we have the right amount of data (68). The +4 is for initial gap, start bit mark and space + stop bit mark.
    if (len(rawbuf) != 2 * NEC_BITS + 4) and (len(rawbuf) != 4):
        print("NEC: Data length=%d is not 68 lr 4" % len(rawbuf))
        return False
    # Check header "mark" this must be done for repeat and data
    if (not match_mark(rawbuf[0], NEC_HEADER_MARK)):
        print("NEC: Header mark length is wrong")
        return False
    # Check command header space
    if (not match_space(rawbuf[1], NEC_HEADER_SPACE)):
        print("NEC: Header space length is wrong")
        return False
    if (not decode_pulse_distance_data(rawbuf[2:], NEC_BITS, NEC_BIT_MARK, NEC_ONE_SPACE, NEC_ZERO_SPACE, PROTOCOL_IS_LSB_FIRST)):
        return False
    # Stop bit
    if (not match_mark(rawbuf[2 + (2 * NEC_BITS)], NEC_BIT_MARK)):
        print("NEC: Stop bit mark length is wrong")
        return False



    
            
#a = np.array(data_nightlight)
a = np.array(data_hitachi)
#show_aeha(a)
decode_nec(a)


