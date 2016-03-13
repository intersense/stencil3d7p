# x,y,z: size of x,y,z dimension
# th_warp: the number of threads in a warp
# maxBW: the max achievable bandwidth of glbal memory
# gltime is Tgpu
def gltime(x,y,z, th_warp, maxBW):
    md = x*y*z*4 >> 20
    if (th_warp <= 32/3):
        time = md/(maxBW*1024*3*th_warp/32) + md*5/(maxBW*1024*1*th_warp/32)
    elif (th_warp <= 32):
        time = md/(maxBW*1024) + md*5/(maxBW*1024*1*th_warp/32)
    elif (th_warp > 32):
        time = md * 6/(maxBW*1024)
    return time

# time of data transfered between CPU and GPU
def gc_time(x,y,z, b):
    md = x*y*z*4 >> 20
    time_cg = md/(b*1024)
    return time_cg


def main():
    for x in [32, 48, 64, 96, 128, 192, 256, 384, 512]:
        for y in [32, 48, 64, 96, 128, 192, 256, 384, 512]:
            for z in [32, 48, 64, 96, 128, 192, 256, 384, 512]:
                #time = gltime(x, y, z, 1 , 180)
                #print("Time of " + str(x) + "x" + str(y) + "x" + str(z) + " floats="+str(time*1000) + "ms")
                t_gpu = gltime(x, y, z, 1, 180)
                t_c2g = gc_time(x, y, z, 5.8)
                t_g2c = gc_time(x, y, z, 6.5)
                t_predict = 0.3232 * t_c2g + t_gpu + 0.6714 * t_g2c
                print("Time of " + str(x) + "x" + str(y) + "x" + str(z) + " floats="+str(t_predict*1000) + "ms")
if __name__ == '__main__':
    main()