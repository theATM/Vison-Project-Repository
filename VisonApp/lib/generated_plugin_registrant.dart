//
// Generated file. Do not edit.
//

// ignore_for_file: lines_longer_than_80_chars

import 'package:soundpool_web/soundpool_web.dart';
import 'package:vibration_web/vibration_web.dart';

import 'package:flutter_web_plugins/flutter_web_plugins.dart';

// ignore: public_member_api_docs
void registerPlugins(Registrar registrar) {
  SoundpoolPlugin.registerWith(registrar);
  VibrationWebPlugin.registerWith(registrar);
  registrar.registerMessageHandler();
}
