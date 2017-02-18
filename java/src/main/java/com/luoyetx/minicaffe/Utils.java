package com.luoyetx.minicaffe;


public final class Utils {
    private Utils() {}
    /**
     * get error message from last C API call
     * @return error message
     */
    public static String GetLastError() {
        return jniGetLastError();
    }
    /**
     * check gpu
     * @return 1 if we can use gpu, else 0
     */
    public static boolean GPUAvailable() {
        if (jniGPUAvailable() == 1) return true;
        else return false;
    }
    /**
     * set caffe runtime device
     * @param mode 1 for gpu, 0 for cpu
     * @param device gpu device if use gpu
     */
    public static void SetCaffeMode(int mode, int device) {
        if (jniSetMode(mode, device) != 0) {
            throw new RuntimeException(GetLastError());
        }
    }
    private static native String jniGetLastError();
    private static native int jniGPUAvailable();
    private static native int jniSetMode(int mode, int device);

    static {
        System.loadLibrary("caffe");
    }
}
