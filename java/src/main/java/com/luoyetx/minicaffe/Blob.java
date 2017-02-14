package com.luoyetx.minicaffe;


/**
 * Blob represent Caffe Blob
 * This class hold the data buffer stand alone in Java float array, process data
 * in Java side and call `syncToC` to copy data back to C side. Same happens when
 * we need to sync data from C side, call `syncToJava`.
 */
public class Blob {
    /**
     * sync C data to Java side, this will update Blob members
     */
    public void syncToJava() {
        if (jniSyncToJava() != 0) {
            throw new RuntimeException(Utils.GetLastError());
        }
    }
    /**
     * sync Java data to C side
     */
    public void syncToC() {
        if (jniSyncToC() != 0) {
            throw new RuntimeException(Utils.GetLastError());
        }
    }
    private native int jniSyncToJava();
    private native int jniSyncToC();
    // internal Blob handle
    private long handle;
    public int num;
    public int channels;
    public int height;
    public int width;
    public float[] data;

    static {
        System.loadLibrary("caffe");
    }
}
