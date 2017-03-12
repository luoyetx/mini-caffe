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
    /**
     * enable profiler
     */
    public static void EnableProfiler() {
        if (jniProfilerEnable() != 0) {
            throw new RuntimeException(GetLastError());
        }
    }
    /**
     * disable profiler
     */
    public static void DisableProfiler() {
        if (jniProfilerDisable() != 0) {
            throw new RuntimeException(GetLastError());
        }
    }
    /**
     * open a scope on profiler
     * @param naem scope name
     */
    public static void OpenScope(String name) {
        if (jniProfilerScopeStart(name) != 0) {
            throw new RuntimeException(GetLastError());
        }
    }
    /**
     * close a scope
     */
    public static void CloseScope() {
        if (jniProfilerScopeEnd() != 0) {
            throw new RuntimeException(GetLastError());
        }
    }
    /**
     * dump profiler data to file
     * @param fn file path
     */
    public static void DumpProfile(String fn) {
        if (jniProfilerDump(fn) != 0) {
            throw new RuntimeException(GetLastError());
        }
    }
    private static native String jniGetLastError();
    private static native int jniGPUAvailable();
    private static native int jniSetMode(int mode, int device);
    private static native int jniProfilerEnable();
    private static native int jniProfilerDisable();
    private static native int jniProfilerScopeStart(String name);
    private static native int jniProfilerScopeEnd();
    private static native int jniProfilerDump(String fn);

    static {
        System.loadLibrary("caffe");
    }
}
