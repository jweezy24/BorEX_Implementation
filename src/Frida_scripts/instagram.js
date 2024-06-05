Java.perform(function() {
    var Bitmap = Java.use("android.graphics.Bitmap");
    var BitmapFactory = Java.use("android.graphics.BitmapFactory");
    var ByteArrayInputStream = Java.use("java.io.ByteArrayInputStream");
    var ByteArrayOutputStream = Java.use("java.io.ByteArrayOutputStream");


    var thread_class = Java.use('X.80q');

    var cached_ml_object = null;

    thread_class.invokeSuspend.implementation = function(k33){
         //Get return of the invoke suspend function.
         
         //Checks to see if the current instance is the ML case.
         if(this.A03.value == 5){
                
            cached_ml_object = this.A02.value;
            send("MODEL CACHED.")
        }

        var ret = this.invokeSuspend(k33);

        return ret;

    }
             

    // Expose a function to be called from Python
    rpc.exports = {
        callpostmessage: function(pixelMatrix) {
            var width = pixelMatrix.length;
            var height = pixelMatrix[0].length; 

            // Access the Bitmap.Config class
            var BitmapConfig = Java.use('android.graphics.Bitmap$Config');

            // Get the ARGB_8888 enum value
            var argb8888Config = BitmapConfig.ARGB_8888.value;

            // Example: Creating a Bitmap using ARGB_8888 config
            var Bitmap = Java.use('android.graphics.Bitmap');
            var bitmap = Bitmap.createBitmap(width, height, argb8888Config);

            // Convert the pixel matrix to a 1D array of ARGB values
            var pixels = [];
            for (var i = 0; i < width; i++) {
                for (var j = 0; j < height; j++) {
                    var pixel = pixelMatrix[i][j];
                    var red = (pixel[0] & 0xFF) << 16;
                    var green = (pixel[1] & 0xFF) << 8;
                    var blue = (pixel[2] & 0xFF);
                    var alpha = 0xFF << 24; // Assuming full opacity
                    var argb = alpha | red | green | blue;
                    pixels.push(argb);
                }
            }

            // Set the pixels on the bitmap
            bitmap.setPixels(pixels, 0, width, 0, 0, width, height);


            if(cached_ml_object == null){
                send("ML model not cached. Execute the model first to continue using this code.");
            }else{

                var clips_xray_obj = Java.cast(cached_ml_object,Java.use("com.instagram.ml.clipsxray.ClipsXRayVisualFeatureExtractor"))
                    

                var thing_7QR = Java.cast(clips_xray_obj.A01.value, Java.use("X.7QR"));

                
                var thing_7OB = Java.use("X.EAY").$new(bitmap);

                var list_thing = Java.use("java.util.Collections").singletonList(thing_7OB);
            
                var thing_G7o =  Java.use("X.7QR").A00(thing_7QR,list_thing);
                

                var list_thing2 = Java.cast(thing_G7o,Java.use("X.EAj")).A00.value;
                
                var ret_string = "";
                for(var k=0; k < list_thing2.size();k++){
                    var thing_7x6 = Java.cast(list_thing2.get(k),Java.use("X.8Lc"));
                    ret_string+= `${thing_7x6.A01.value},${thing_7x6.A00.value} \t`;
                }
                send(ret_string);
            }


        }
    };
});