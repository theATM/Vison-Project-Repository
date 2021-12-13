import 'dart:async';
import 'dart:core';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:vibration/vibration.dart';
import 'package:flutter_flashlight/flutter_flashlight.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:wakelock/wakelock.dart';




Future<void> main() async {
  // Ensure that plugin services are initialized so that `availableCameras()`
  // can be called before `runApp()`
  WidgetsFlutterBinding.ensureInitialized();

  // Obtain a list of the available cameras on the device.
  final cameras = await availableCameras();

  // Get a specific camera from the list of available cameras.
  final firstCamera = cameras[0];
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
  bool isTapped = false;
  bool cameraPermission = false;
  CameraController _controller;
  String error;
  Future<void> _initializeControllerFuture;
  static const platform = const MethodChannel('samples.flutter.dev/battery');
  int _prediction = 0;
  bool busy = false;
  bool alreadyChecked = false;
  // displaying values
  List<String> announcements = ["10 PLN", "20 PLN", "50 PLN", "100 PLN", "200 PLN", "500 PLN", ""];
  List<String> announcMore = [
    "Skieruj kamerę na banknot",
    "Point the camera at the banknote",
    "zamknąć aplikację",
    "close the application",
    "Zamykam aplikację",
    "Closing the application"
  ];
  String currentDisplay = " ";
  int langAdd = 0;
  List<String> money = ["10", "20", "50", "100", "200", "500", "None"];
  List<int> _currentPredictionsList = new List.generate(8, (int index) => 0);
  int maxCorrectPredictionsInSequence =
  3; // this is arbitrary and can be changed.
  bool hasDetectedPrediction = false;
  int _NoneIter = 40;
  bool firstStart = true;
  // vibration
  bool hasVibration = false;
  bool hasCustomVibrationsSupport = false;
  String lang = Platform.localeName;
  // flashlight
  bool hasFlash = false;
  bool flashlight = false;
  int _flashIter = 0;
  bool hasScreenReader = false;

  void updatePredictionAction() async {
    if (this.money[_prediction] == 'None') {
      currentDisplay = " ";
      if(_NoneIter > 0){
        _NoneIter = _NoneIter - 1;
      } else {
        SemanticsService.announce(announcMore[0+langAdd], TextDirection.ltr);
        if (this.hasVibration && this.hasCustomVibrationsSupport) {
          Vibration.vibrate(pattern: [0, 150]);
        }
        _NoneIter = 40;
      }
    }

    // if has detected prediction previously, do nothing.
    if (this.hasDetectedPrediction) {
      return;
    }

    if (this.money[_prediction] != 'None') {
      _NoneIter = 40;
      currentDisplay = this.money[_prediction];
      SemanticsService.announce(announcements[_prediction], TextDirection.ltr);
      if (this.hasVibration && this.hasCustomVibrationsSupport) {
        Vibration.vibrate(pattern: [0, 150]);
      }
    }
  }

  Future<void> _getPrediction(CameraImage cameraImage) async {
    var framesY = cameraImage.planes[0].bytes;
    var framesU = cameraImage.planes[1].bytes;
    var framesV = cameraImage.planes[2].bytes;

    double yValue = 0.0;

    yValue = framesY.map((e) => e).reduce((yValue, element) => yValue + element) / framesY.length;

    debugPrint('Y mean:  $yValue');

    if(this.hasFlash){  // Jeśli można użyć lampy
      if (_flashIter == 0 && flashlight == true){
        flashlight = false;
      } else if(yValue < 20 && flashlight == false){
        flashlight = true;
        _flashIter = 15;
      } else if(yValue > 70 && flashlight == true) {
        _flashIter = _flashIter - 1;
      }
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
    this.hasFlash = await Flashlight.hasFlashlight;
  }

  void checkPermissionOptions() async {
    this.cameraPermission = await Permission.camera.isGranted;
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
        ResolutionPreset.high,
        enableAudio: false
    );
    // Next, initialize the controller. This returns a Future.
    _initializeControllerFuture = _controller.initialize();
    Wakelock.enable();

    this.checkDeviceVibrationOptions();
    this.checkFlashlightOptions();
    this.checkPermissionOptions();

    if(lang != 'pl_PL') langAdd = 1;

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
          this._currentPredictionsList.clear();
          this.hasDetectedPrediction = false;
        }

        if (this._currentPredictionsList.length >=
            this.maxCorrectPredictionsInSequence) {
          this._prediction = this._currentPredictionsList.last;
          this.updatePredictionAction();
          this.hasDetectedPrediction = true;
        }

        if (this.flashlight) {
          _controller.setFlashMode(FlashMode.torch);
        } else _controller.setFlashMode(FlashMode.off);

        setState(() {
          busy = false;
        });
      }
      return true;
    });

    _initializeControllerFuture.then((x) {
      _controller.startImageStream((CameraImage availableImage) async {
        if (busy) {
          return;
        }
        debugPrint(_controller.value.aspectRatio.toString());
        await _getPrediction(availableImage);
      });
    });
  }

  @override
  void dispose() {
    // Dispose of the controller when the widget is disposed.
    WidgetsBinding.instance.removeObserver(this);
    _controller.stopImageStream();
    _controller.setFlashMode(FlashMode.off);
    _controller.dispose();
    Wakelock.disable();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) async{
    cameraPermission = await Permission.camera.isGranted;
    if (state == AppLifecycleState.detached ||
        state == AppLifecycleState.paused ||
        (state == AppLifecycleState.inactive && cameraPermission == true) ||
        (state == AppLifecycleState.resumed && cameraPermission == false)){
      this.dispose();
      exit(0);
    }
  }

  @override
  Widget build(BuildContext context) {
    if(cameraPermission && !alreadyChecked) {
      SemanticsService.announce(announcMore[0+langAdd], TextDirection.ltr);
      alreadyChecked = true;
    }
    final mediaQueryData = MediaQuery.of(context);
    if(mediaQueryData.accessibleNavigation) {
      return Semantics(
        onTapHint: announcMore[2+langAdd],
        child: Material(
          child: Container(
            child: GestureDetector(
              child: FittedBox(
                fit: BoxFit.fill,
                child:Semantics(
                  excludeSemantics: true,
                  child: Text(
                    currentDisplay,
                    style: TextStyle(
                      color: Colors.limeAccent,
                      backgroundColor: Colors.black,
                      fontSize: 10000,
                    ),
                  ),
                ),
              ),
              onTap: () {
                if (!isTapped) {
                  this.isTapped = true;
                  SemanticsService.announce(announcMore[4+langAdd], TextDirection.ltr);
                  Timer(Duration(milliseconds: 500), () {
                    this.dispose();
                    exit(0);
                  });
                }
              },
            ),
          ),
        ),
      );
    } else {
      return Stack(
        children: [
          Center  (
            child: Container (
              child: new CameraPreview(_controller),
            ),
          ),
          Center(
            child: Material(
              type: MaterialType.transparency,
              child: Container(
                  child: FittedBox(
                    fit: BoxFit.fill,
                    child:Text(
                      currentDisplay,
                      style: TextStyle(
                        color: Colors.limeAccent,
                        fontSize: 10000,
                      ),
                    ),
                  )
              ),
            ),
          )
        ],
      );
    }
  }
}


