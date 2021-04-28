import 'dart:async';
import 'dart:core';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:vibration/vibration.dart';
import 'package:soundpool/soundpool.dart';

Future<void> main() async {
  // Ensure that plugin services are initialized so that `availableCameras()`
  // can be called before `runApp()`
  WidgetsFlutterBinding.ensureInitialized();

  // Obtain a list of the available cameras on the device.
  final cameras = await availableCameras();

  // Get a specific camera from the list of available cameras.
  final firstCamera = cameras.first;

  runApp(
    MaterialApp(
      theme: ThemeData.dark(),
      home: TakePictureScreen(
        // Pass the appropriate camera to the TakePictureScreen widget.
        camera: firstCamera,
      ),
    ),
  );
}

// A screen that allows users to take a picture using a given camera.
class TakePictureScreen extends StatefulWidget {
  final CameraDescription camera;

  const TakePictureScreen({
    @required this.camera,
  }) : super();

  @override
  TakePictureScreenState createState() => TakePictureScreenState();
}

class TakePictureScreenState extends State<TakePictureScreen> {
  CameraController _controller;
  Soundpool soundpool;
  String error;
  Future<void> _initializeControllerFuture;
  int frames = 0;
  static const platform = const MethodChannel('samples.flutter.dev/battery');
  int frameCounter = 0;
  int _prediction = 0;
  int _iter = 0;
  bool busy = false;
  // displaying values
  List<String> money = ["10", "20", "50", "100", "200", "500", "None"];
  List<int> _currentPredictionsList = new List.generate(8, (int index) => 0);
  int maxCorrectPredictionsInSequence =
      4; // this is arbitrary and can be changed.
  String defaultDisplayMessage = "Skieruj kamerę na banknot!";
  String currentDisplayMessage = "";
  bool hasDetectedPrediction = false;
  // vibration
  bool hasVibration = false;
  bool hasCustomVibrationsSupport = false;
  Map vibrationmap = new Map();
  // sound
  Map soundmap = new Map();

  void predictionActions(String value) async {
    // play sound depending on value.
    await this.soundpool.play(this.soundmap[value]);
    // vibrate depending on value.
    if (this.hasVibration && this.hasCustomVibrationsSupport) {
      Vibration.vibrate(pattern: this.vibrationmap[value]);
    }
  }

  void updatePredictionMessage(int prediction) async {
    // if has detected prediction previously, do nothing.
    if (this.hasDetectedPrediction) {
      return;
    }

    if (this.money[_prediction] == 'None') {
      this.currentDisplayMessage = defaultDisplayMessage;
    } else {
      predictionActions(this.money[_prediction]);
      setState(() {
        this.currentDisplayMessage = "Banknot " + money[_prediction] + " złoty";
      });
    }
  }

  Future<void> _getPrediction(CameraImage cameraImage) async {
    var framesY = cameraImage.planes[0].bytes;
    var framesU = cameraImage.planes[1].bytes;
    var framesV = cameraImage.planes[2].bytes;

    double yValue = 0.0;
    //double uValue = 0.0;
    //double vValue = 0.0;

    yValue = framesY.map((e) => e).reduce((yValue, element) => yValue + element) / framesY.length;
    //uValue = framesU.map((e) => e).reduce((uValue, element) => uValue + element) / framesU.length;
    //vValue = framesV.map((e) => e).reduce((vValue, element) => vValue + element) / framesV.length;

    debugPrint('Y mean:  $yValue');

    if (yValue < 15) {
      setState(() {
        busy = true;
      });
      await Future.delayed(Duration(seconds: 1));
      setState(() {
        busy = false;
      });
      return;
    }

    try {
      platform.invokeMethod('getPrediction', <String, dynamic>{
        'width': cameraImage.width,
        'height': cameraImage.height,
        'Y': framesY,
        'U': framesU,
        'V': framesV
      });
    } on PlatformException catch (e) {
      print("error: ${e.message}");
      error = e.message;
    }

    setState(() {
      busy = true;
    });
  }

  void checkDeviceVibrationOptions() async {
    if (await Vibration.hasVibrator()) {
      this.hasVibration = true;
      if (await Vibration.hasCustomVibrationsSupport()) {
        this.hasCustomVibrationsSupport = true;
      }
    }
  }

  void loadSoundAssets() async {
    this.soundpool = Soundpool(streamType: StreamType.music);
    ByteData asset;
    asset = await rootBundle.load("sounds/10.wav");
    this.soundmap["10"] = await soundpool.load(asset);
    asset = await rootBundle.load("sounds/20.wav");
    this.soundmap["20"] = await soundpool.load(asset);
    asset = await rootBundle.load("sounds/50.wav");
    this.soundmap["50"] = await soundpool.load(asset);
    asset = await rootBundle.load("sounds/100.wav");
    this.soundmap["100"] = await soundpool.load(asset);
    asset = await rootBundle.load("sounds/200.wav");
    this.soundmap["200"] = await soundpool.load(asset);
    asset = await rootBundle.load("sounds/500.wav");
    this.soundmap["500"] = await soundpool.load(asset);
    asset = await rootBundle.load("sounds/None.wav");
    this.soundmap["None"] = await soundpool.load(asset);
    // play starting sound
    await this.soundpool.play(this.soundmap['None']);
  }

  void loadVibrationOptions() {
    // in millis.
    this.vibrationmap["10"] = [0, 300];
    this.vibrationmap["20"] = [0, 300, 300, 300];
    this.vibrationmap["50"] = [0, 300, 300, 300, 300, 300];
    this.vibrationmap["100"] = [0, 1000, 300, 1000];
    this.vibrationmap["200"] = [0, 1000, 300, 1000, 300, 1000];
    this.vibrationmap["500"] = [0, 3000];
  }

  @override
  void initState() {
    super.initState();
    // To display the current output from the Camera,
    // create a CameraController.
    _controller = CameraController(
      // Get a specific camera from the list of available cameras.
      widget.camera,
      // Define the resolution to use.
      ResolutionPreset.max,
    );
    // Next, initialize the controller. This returns a Future.
    _initializeControllerFuture = _controller.initialize();

    this.checkDeviceVibrationOptions();
    this.currentDisplayMessage = defaultDisplayMessage;
    this.loadSoundAssets();
    this.loadVibrationOptions();

    platform.setMethodCallHandler((MethodCall call) async {
      if (call.method == "predictionResult") {
        final args = call.arguments;
        int currentPrediction = args["result"];
        if (this._currentPredictionsList.isEmpty) {
          debugPrint('predictions list empty');
          this._currentPredictionsList.add(currentPrediction);
        } else if (this._currentPredictionsList.contains(currentPrediction)) {
          debugPrint('predictions list matching previous');
          this._currentPredictionsList.add(currentPrediction);
        } else {
          debugPrint('Clear predictions list');
          String lastValue = money[currentPrediction];
          debugPrint('Last value: $lastValue');
          this._currentPredictionsList.clear();
          debugPrint('Size of list: +' +
              this._currentPredictionsList.length.toString());
          this.hasDetectedPrediction = false;
        }

        if (this._currentPredictionsList.length >=
            this.maxCorrectPredictionsInSequence) {
          this._prediction = this._currentPredictionsList.last;
          this.updatePredictionMessage(this._prediction);
          this.hasDetectedPrediction = true;
        }

        setState(() {
          _iter = _iter + 1;
          busy = false;
        });
      }
      return true;
    });

    _initializeControllerFuture.then((x) {
      _controller.startImageStream((CameraImage availableImage) async {
        if (busy) return;
        frames = 0;
        await _getPrediction(availableImage);
      });
    });
  }

  @override
  void dispose() {
    // Dispose of the controller when the widget is disposed.
    _controller.stopImageStream();
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
          title: Text(
        this.currentDisplayMessage,
        textScaleFactor: 1.4,
      )),
      // Wait until the controller is initialized before displaying the
      // camera preview. Use a FutureBuilder to display a loading spinner
      // until the controller has finished initializing.
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            // If the Future is complete, display the preview.
            return CameraPreview(_controller);
          } else {
            // Otherwise, display a loading indicator.
            return Center(child: CircularProgressIndicator());
          }
        },
      ),
    );
  }
}

// A widget that displays the picture taken by the user.
class DisplayPictureScreen extends StatelessWidget {
  final String imagePath;

  const DisplayPictureScreen({Key key, @required this.imagePath})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Display the Picture')),
      // The image is stored as a file on the device. Use the `Image.file`
      // constructor with the given path to display the image.
      body: Image.file(File(imagePath)),
    );
  }
}
