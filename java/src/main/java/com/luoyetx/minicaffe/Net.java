package com.luoyetx.minicaffe;


/**
 * Net represent Caffe Net object
 * Typical usage is creating a Net, get input blobs through `getBlob`,
 * then fill data to these blobs, call `Blob.syncToC` make network internal
 * buffer update to blobs data. Call `forward` to forward network, then get
 * output blobs to get network output data. Note that `forward` also may
 * affect network internal blobs which makes blobs in Java side out of date.
 */
public final class Net {
    /**
     * construct Net by given network path
     * @param net_path network prototxt path
     * @param model_path network caffemodel path
     */
    public Net(String net_path, String model_path) {
        if (jniCreate(net_path, model_path) != 0) {
            throw new RuntimeException(Utils.GetLastError());
        }
    }
    /**
     * construct Net by given network buffer
     * @param net_buffer network prototxt buffer
     * @param model_buffer network caffemodel buffer
     */
    public Net(byte[] net_buffer, byte[] model_buffer) {
        if (jniCreateFromBuffer(net_buffer, model_buffer) != 0) {
            throw new RuntimeException(Utils.GetLastError());
        }
    }
    @Override
    public void finalize() {
        if (jniDestroy() != 0) {
            throw new RuntimeException(Utils.GetLastError());
        }
    }
    /**
     * mark network internal blob as output
     * mark inside blob if you need to use it and it is not an output blob,
     * otherwise you may get the wrong result
     * @param name blob name
     */
    public void markOutput(String name) {
        if (jniMarkOutput(name) != 0) {
            throw new RuntimeException(Utils.GetLastError());
        }
    }
    /**
     * forward the network
     * this function may make blobs in Java side out of date,
     * call `Blob.syncToJava` if need
     */
    public void forward() {
        if (jniForward() != 0) {
            throw new RuntimeException(Utils.GetLastError());
        }
    }
    /**
     * get blob by name
     * @param name blob name in network data buffers
     * @return blob
     */
    public Blob getBlob(String name) {
        Blob blob = new Blob();
        if (jniGetBlob(name, blob) != 0) {
            throw new RuntimeException(Utils.GetLastError());
        }
        return blob;
    }
    private native int jniCreate(String net_path, String model_path);
    private native int jniCreateFromBuffer(byte[] net_buffer, byte[] model_buffer);
    private native int jniDestroy();
    private native int jniMarkOutput(String name);
    private native int jniForward();
    private native int jniGetBlob(String name, Blob blob);
    // internal Net handle
    private long handle;

    static {
        System.loadLibrary("caffe");
    }
}
