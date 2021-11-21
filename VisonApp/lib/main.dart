import 'dart:async';
import 'dart:core';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:vibration/vibration.dart';
import 'package:soundpool/soundpool.dart';
import 'package:flutter_flashlight/flutter_flashlight.dart';




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

class TakePictureScreenState extends State<TakePictureScreen> with WidgetsBindingObserver {
  AppLifecycleState _lastLifecycleState = null;
  bool isTapped = false;
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
  int _NoneIter = 40;
  // flashlight
  bool hasFlash = false;
  bool flashlight = false;
  int _flashIter = 0;

  void predictionActions(String value) async {
    // play sound depending on value.
    await this.soundpool.play(this.soundmap[value]);
    // vibrate depending on value.
    if (this.hasVibration && this.hasCustomVibrationsSupport) {
      Vibration.vibrate(pattern: this.vibrationmap[value]);
    }
  }

  void updatePredictionAction(int prediction) async {

    if (this.money[prediction] == 'None') {
      if(_NoneIter > 0){
        _NoneIter = _NoneIter - 1;
      } else {
        await this.soundpool.play(this.soundmap['None']);
        if (this.hasVibration && this.hasCustomVibrationsSupport) {
          Vibration.vibrate(pattern: this.vibrationmap['None']);
        }
        _NoneIter = 40;
      }
      debugPrint('NoneIter: '+_NoneIter.toString());
    }

    // if has detected prediction previously, do nothing.
    if (this.hasDetectedPrediction) {
      return;
    }

    if (this.money[prediction] != 'None') {
      predictionActions(this.money[prediction]);
    }

  }

  void flashlightAction(double value){
    // if this device has flashlight and it is not completely dark
    if(value < 30 && flashlight == false){
      flashlight = true;
      _flashIter = 20;
    } else if(value > 70 && flashlight == true){
      _flashIter = _flashIter - 1;
    }

    if(_flashIter == 0 && flashlight == true){
      flashlight == false;
    }
  }

  Future<void> _getPrediction(CameraImage cameraImage) async {
    var framesY = cameraImage.planes[0].bytes;
    var framesU = cameraImage.planes[1].bytes;
    var framesV = cameraImage.planes[2].bytes;

    double yValue = 0.0;

    yValue = framesY.map((e) => e).reduce((yValue, element) => yValue + element) / framesY.length;

    debugPrint('Y mean:  $yValue');

    // if there is completely dark, wait 1 second and try with another input
    if (yValue < 10) {
      setState(() {
        busy = true;
        flashlight = false;
      });
      await Future.delayed(Duration(seconds: 1));
      setState(() {
        busy = false;
      });
      return;
    } else if (this.hasFlash) {
      flashlightAction(yValue);
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

  void checkFlashlightOptions() async {
    if (await Flashlight.hasFlashlight){
      this.hasFlash = true;
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
    asset = await rootBundle.load("sounds/End.wav");
    this.soundmap["End"] = await soundpool.load(asset);
    // play starting sound
    await this.soundpool.play(this.soundmap['None']);
  }

  void loadVibrationOptions() {
    // in millis.
    this.vibrationmap["None"] = [0, 150, 150];
    this.vibrationmap["10"] = [0, 300];
    this.vibrationmap["20"] = [0, 300, 300, 300];
    this.vibrationmap["50"] = [0, 300, 300, 300, 300, 300];
    this.vibrationmap["100"] = [0, 1000, 300, 1000];
    this.vibrationmap["200"] = [0, 1000, 300, 1000, 300, 1000];
    this.vibrationmap["500"] = [0, 3000];
  }

  @override
  void initState() {
    debugPrint('Starting....');
    super.initState();
    WidgetsBinding.instance.addObserver(this);
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
    this.loadSoundAssets();
    this.loadVibrationOptions();
    this.checkFlashlightOptions();

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
          this.updatePredictionAction(this._prediction);
          this.hasDetectedPrediction = true;
        }

        if (this.flashlight) {
          _controller.setFlashMode(FlashMode.torch);
        } else _controller.setFlashMode(FlashMode.off);

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
    WidgetsBinding.instance.removeObserver(this);
    _controller.stopImageStream();
    _controller.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    setState(() {
      _lastLifecycleState = state;
    });
  }

  @override
  Widget build(BuildContext context) {

    if ( _lastLifecycleState != null) {
      this.dispose();
      exit(0);
    }

    return Scaffold(
      // Wait until the controller is initialized before displaying the
      // camera preview. Use a FutureBuilder to display a loading spinner
      // until the controller has finished initializing.
        body: Container(
            color: Colors.black54,
            child: GestureDetector(
              onDoubleTap: (){
                if (!isTapped) {
                  this.isTapped = true;
                  this.soundpool.play((this.soundmap['End']));
                  Timer(Duration(seconds: 2), () {
                    this.dispose();
                    exit(0);
                  }
                  );
                }
              },
            )
        )
    );
  }
}


