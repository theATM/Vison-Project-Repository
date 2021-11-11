package vison.visontestapp;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity
{
    Button nutton;
    TextView text;

    Module module;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        this.text = (TextView) findViewById(R.id.vvv);
        this.nutton = (Button) findViewById(R.id.button);
        this.nutton.setOnClickListener(myOnClickListener);
    }


    View.OnClickListener myOnClickListener = new View.OnClickListener()
    {
        @Override
        public void onClick(View view)
        {
            //I clikced my button
            MainActivity.this.text.setText("Start");
            Log.d("Tutaj Olek","in");
            MainActivity.this.module = getModel("test1243.pt");
            int result = getInferenceResults(createMockInput());
        }
    };


    public Module getModel(String name)
    {
        Module m = null;
        try
        {
            m = Module.load(assetFilePath(this, name));
        }
        catch(Exception e)
        {
            Log.d("Exception", e.toString());
            e.printStackTrace();
            return m;
        }
        return m;
    }

    public static String assetFilePath(Context context, String assetName) throws IOException
    {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0)
        {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName))
        {
            try (OutputStream os = new FileOutputStream(file))
            {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1)
                {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            catch (Exception e)
            {
                e.printStackTrace();
            }
            return file.getAbsolutePath();
        }
        catch (Exception e)
        {
            Log.d("Exception", "File reading" + e.toString());
            e.printStackTrace();
            return "";
        }
    }

    private int getInferenceResults(ByteArrayOutputStream binaryStream)
    {
        long start = System.currentTimeMillis();
        ByteArrayInputStream istream = new ByteArrayInputStream(binaryStream.toByteArray());


        Bitmap bitmap = BitmapFactory.decodeStream(istream);
        bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, false);

        final float[] meanArray = {0.48269427f, 0.43759444f, 0.4045701f};
        final float[] stdArray = {0.24467267f, 0.23742135f, 0.24701703f};
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, meanArray, stdArray);

        long stop = System.currentTimeMillis();
        long time = stop - start;

        Log.d("Time", "Convert bitmap to float tensor: " +time);


        Tensor outputTensor = this.module.forward(IValue.from(inputTensor)).toTensor();
        long sxd = System.currentTimeMillis();
        time = sxd - stop;


        Log.d("Time", "Inference: " + time);



        float[] scores = outputTensor.getDataAsFloatArray();

        int maxInd = -1;
        float max = -999999.0f;
        for (int i = 0; i < scores.length; i++)
        {
            if (max < scores[i])
            {
                max = scores[i];
                maxInd = i;
            }

            Log.d("SCORES", String.format("Class: %d -----  %f", i, scores[i]));

        }

        Log.d("Prediction", String.format("Class: %d -----  %f", maxInd, scores[maxInd]));

        return maxInd;
    }

    private ByteArrayOutputStream createMockInput()
    {
        ByteArrayOutputStream bs = new ByteArrayOutputStream();


        return bs;
    }





}