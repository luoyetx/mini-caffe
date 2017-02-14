package com.luoyetx.minicaffe;

/**
 * Net represent Caffe Net object
 */
public class Net {
    /**
     * construct Net by given parameters
     * @param net_path network prototxt path
     * @param model_path network caffemodel path
     */
    Net(String net_path, String model_path) {
        if (CCreate(net_path, model_path) != 0) {
            throw new RuntimeException(Utils.GetLastError());
        }
    }
    @Override
    public void finalize() {
        if (CDestroy() != 0) {
            throw new RuntimeException(Utils.GetLastError());
        }
    }
    /**
     * forward the network
     */
    public void forward() {
        if (CForward() != 0) {
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
        if (CGetBlob(name, blob) != 0) {
            throw new RuntimeException(Utils.GetLastError());
        }
        return blob;
    }
    private native int CCreate(String net_path, String model_path);
    private native int CDestroy();
    private native int CForward();
    private native int CGetBlob(String name, Blob blob);
    // internal Net handle
    private long handle;
}
