package com.example.flutterapp;

import android.content.Context;
import android.graphics.*;
import android.os.Handler;
import android.os.Looper;
import androidx.annotation.NonNull;
import io.flutter.Log;
import io.flutter.embedding.android.FlutterActivity;
import io.flutter.embedding.engine.FlutterEngine;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugins.GeneratedPluginRegistrant;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import java.io.*;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.lang.Integer;
import java.util.HashMap;
import android.os.Bundle;

public class MainActivity extends FlutterActivity {
    private static final boolean isDebug = false; 
    private static final String CHANNEL = "samples.flutter.dev/battery";
    private static FlutterEngine _flutterEngine;
    private static final LinkedBlockingQueue<Runnable> neuralNetFrameQueue = new LinkedBlockingQueue<Runnable>(1);
    Module module;
    private static final ThreadPoolExecutor neuralNetThreadPool = new ThreadPoolExecutor(
            1,       // Initial pool size
            1,       // Max pool size
            10,
            java.util.concurrent.TimeUnit.SECONDS,
            neuralNetFrameQueue);

    @Override
    protected void onDestroy() {
        _flutterEngine.getPlatformViewsController().detachFromView();
        super.onDestroy();
    }

    public void onCreate(Bundle savedState)
    {
        super.onCreate(savedState);
        module = getModel("resnet_18_acc94_29.pt");
    }

    private MethodChannel flutterChannel;

    @Override
    public void configureFlutterEngine(@NonNull FlutterEngine flutterEngine) {
        super.configureFlutterEngine(flutterEngine);
        _flutterEngine = flutterEngine; 
        flutterChannel = new MethodChannel(flutterEngine.getDartExecutor().getBinaryMessenger(), CHANNEL);
        flutterChannel.setMethodCallHandler(
                        (call, result) -> {
                            if (call.method.equals("getPrediction")) {
                                try {
                                    neuralNetThreadPool.execute(() -> {
                                        long start = System.currentTimeMillis();
                                        byte[] plateY = call.argument("Y");
                                        byte[] plateU = call.argument("U");
                                        byte[] plateV = call.argument("V");
                                        int width = call.argument("width");
                                        int height = call.argument("height");

                                        byte[] yuv = new byte[plateY.length + plateV.length + plateU.length];
                                        System.arraycopy(plateY, 0, yuv, 0, plateY.length);
                                        System.arraycopy(plateV, 0, yuv, plateY.length, plateV.length);
                                        System.arraycopy(plateU, 0, yuv, plateY.length + plateV.length, plateU.length);
                                        YuvImage x = new YuvImage(yuv, ImageFormat.NV21, width, height, null);

                                        int x1, x2, y1, y2;
                                        if (width > height) {
                                            x1 = (width - height) / 2;
                                            x2 = x1 + height;
                                            y1 = 0;
                                            y2 = height;
                                        } else {
                                            x1 = 0;
                                            x2 = width;
                                            y1 = (height - width) / 2;
                                            y2 = y1 + width;
                                        }
                                        Rect r = new Rect(x1, y1, x2, y2);
                                        ByteArrayOutputStream bs = new ByteArrayOutputStream();
                                        x.compressToJpeg(r, 100, bs);

                                        int res = getInferenceResults(bs);

                                        // Wysłanie wyników do Fluttera
                                        new Handler(Looper.getMainLooper()).post(() -> {
                                            flutterChannel.invokeMethod("predictionResult", new HashMap<String, Integer>() {{
                                                put("result", res);
                                            }});
                                        });
                                    });
                                } catch (Exception e) {
                                    e.printStackTrace();
                                    result.success(false);
                                    return;
                                }
                            } else {
                                result.notImplemented();
                                return;
                            }
                            result.success(true);
                        }
                );
    };


    private int getInferenceResults(ByteArrayOutputStream binaryStream) {
        long start = System.currentTimeMillis();
        ByteArrayInputStream istream = new ByteArrayInputStream(binaryStream.toByteArray());


        Bitmap bitmap = BitmapFactory.decodeStream(istream);
        bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, false);

        final float[] meanArray = {0.48269427f, 0.43759444f, 0.4045701f};
        final float[] stdArray = {0.24467267f, 0.23742135f, 0.24701703f};
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, meanArray, stdArray);

        long stop = System.currentTimeMillis();
        long time = stop - start;
        if(isDebug) {
            Log.d("Time", "Convert bitmap to float tensor: " +time);
        }

        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        long sxd = System.currentTimeMillis();
        time = sxd - stop;

        if(isDebug) {
            Log.d("Time", "Inference: " + time);
        }

        float[] scores = outputTensor.getDataAsFloatArray();

        int maxInd = -1;
        float max = -999999.0f;
        for (int i = 0; i < scores.length; i++) {
            if (max < scores[i]) {
                max = scores[i];
                maxInd = i;
            }
            if(isDebug) {
                Log.d("SCORES", String.format("Class: %d -----  %f", i, scores[i]));
            }
        }
        if(isDebug) {
            Log.d("Prediction", String.format("Class: %d -----  %f", maxInd, scores[maxInd]));
        }
        return maxInd;
    }

    public Module getModel(String name) {
        Module m = null;
        try {
            m = Module.load(assetFilePath(this, name));
        } catch(Exception e){
            Log.d("Exception", e.toString());
            e.printStackTrace();
            return m;
        }
        return m;
    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            } catch(Exception e){
                e.printStackTrace();
            }
            return file.getAbsolutePath();
        } catch (Exception e){
            Log.d("Exception", "File reading" + e.toString());
            e.printStackTrace();
            return "";
        }
    }
}


