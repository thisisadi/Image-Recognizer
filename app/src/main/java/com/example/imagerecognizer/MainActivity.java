package com.example.imagerecognizer;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.imagerecognizer.ml.ImageRecognizerModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    TextView result;
    ImageView imageView;
    Button select,predict;
    Bitmap bitmap;
    int imageSize = 224;
    List<String> values;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getSupportActionBar().hide();
        // To ensure statusBar has same background as that of the activity.
        Window w = getWindow();
        w.setFlags(WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS, WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS);
        result = findViewById(R.id.textView);
        select = findViewById(R.id.select);
        predict = findViewById(R.id.predict);
        imageView = findViewById(R.id.imageView);
        imageView.setVisibility(View.GONE);
        result.setVisibility(View.GONE);
        // Lines 57 to 71 are used to read the "labels.txt" file and convert it into a string array of all the items present in the file.
        String fileName = "labels.txt";
        try {
            BufferedReader bReader = new BufferedReader(new InputStreamReader(getAssets().open(fileName)));
             values = new ArrayList<>();
            String line = bReader.readLine();
            while (line != null) {
                values.add(line);
                line = bReader.readLine();
            }
            bReader.close();
            for (String v : values)
                Log.i("Array is ", v);
        } catch (IOException e) {
            e.printStackTrace();
        }

        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Used for asking permission from the user to click a photo using the Camera.
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });

        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (bitmap==null){
                    Toast.makeText(getApplicationContext(), "Please click a Pic!", Toast.LENGTH_SHORT).show();
                }
                else{
                    try {
                        ImageRecognizerModel model = ImageRecognizerModel.newInstance(MainActivity.this);

                        // Creates inputs for reference.
                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8);
                        TensorImage tBuffer = TensorImage.fromBitmap(bitmap);
                        ByteBuffer byteBuffer = tBuffer.getBuffer();
                        inputFeature0.loadBuffer(byteBuffer);

                        // Runs model inference and gets result.
                        ImageRecognizerModel.Outputs outputs = model.process(inputFeature0);
                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer(); /* Contains the probabilities of each item in the "labels.txt" file
                                                                                                of being the same item as that in the clicked picture */

                        int index = getMax(outputFeature0.getFloatArray()); /* Getting the index of the highest probability item and displaying it as the clicked picture
                                                                         as a prediction of the ML model */
                        result.setText(values.get(index));
                        result.setVisibility(View.VISIBLE);

                        // Releases model resources if no longer used.
                        model.close();
                    } catch (IOException e) {
                    }
                }
            }
        });
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 1 && resultCode == RESULT_OK) {
            bitmap = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(bitmap.getWidth(),bitmap.getHeight());
            bitmap = ThumbnailUtils.extractThumbnail(bitmap,dimension,dimension); // Lines 125 to 129 perform a 'Center Crop' on the clicked picture and set it as imageView.
            imageView.setImageBitmap(bitmap);
            imageView.setVisibility(View.VISIBLE);
            bitmap = Bitmap.createScaledBitmap(bitmap,imageSize,imageSize,true); /* This is responsible for converting the bitmap into a bitmap of dimensions that the
                                                                                        Ml model will accept as input */
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
    /* Function to get the index of the item present in the "labels.txt" file (which has been converted into a string array), which has the highest probability of being the
    same item as that in the clicked picture */
    public int getMax(float[] arr){
        int ind = 0;
        float max = 0.0f;
        for (int i = 0; i < 1000; i++){
            if (arr[i] > max){
                ind = i;
                max = arr[i];
            }
        }
        return ind;
    }
}