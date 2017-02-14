package com.luoyetx.minicaffe;


/**
 * Net represent Caffe Net object
 * Typical usage is creating a Net, get input blobs through `getBlob`,
 * then fill data to these blobs, call `Blob.syncToC` make network internal
 * buffer update to blobs data. Call `forward` to forward network, then get
 * output blobs to get network output data. Note that `forward` also may
 * affect network internal blobs which makes blobs in Java side out of date.
 */
public class Net {
    /**
     * construct Net by given parameters
     * @param net_path network prototxt path
     * @param model_path network caffemodel path
     */
    public Net(String net_path, String model_path) {
        if (jniCreate(net_path, model_path) != 0) {
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
    private native int jniDestroy();
    private native int jniForward();
    private native int jniGetBlob(String name, Blob blob);
    // internal Net handle
    private long handle;

    static {
        System.loadLibrary("caffe");
    }
}
