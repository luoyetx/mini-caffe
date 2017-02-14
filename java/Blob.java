package com.luoyetx.minicaffe;


/**
 * Blob represent Caffe Blob
 * This class hold the data buffer stand alone in Java float array, process data
 * in Java side and call `syncToC` to copy data back to C side. Same happens when
 * we need to sync data from C side, call `syncToJava`.
 */
public class Blob {
    public Blob() {
        handle = 0;
        num = channels = height = width = 0;
        data = new float[];
    }
    /**
     * sync C data to Java side, this will update Blob members
     */
    public void syncToJava() {
        if (CSyncToJava() != 0) {
            throw new RuntimeException(Utils.GetLastError());
        }
    }
    /**
     * sync Java data to C side
     */
    public void syncToC() {
        if (CSyncToC() != 0) {
            throw new RuntimeException(Utils.GetLastError());
        }
    }
    private native int CSyncToJava();
    private native int CSyncToC();
    // internal Blob handle
    private long handle;
    public int num;
    public int channels;
    public int height;
    public int width;
    public float[] data;
}
