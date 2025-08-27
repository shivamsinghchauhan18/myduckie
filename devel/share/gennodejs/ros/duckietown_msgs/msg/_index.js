
"use strict";

let WheelEncoderStamped = require('./WheelEncoderStamped.js');
let TagInfo = require('./TagInfo.js');
let IntersectionPoseImg = require('./IntersectionPoseImg.js');
let DiagnosticsRosTopic = require('./DiagnosticsRosTopic.js');
let DiagnosticsRosProfilingUnit = require('./DiagnosticsRosProfilingUnit.js');
let SceneSegments = require('./SceneSegments.js');
let Trajectory = require('./Trajectory.js');
let FSMState = require('./FSMState.js');
let NodeParameter = require('./NodeParameter.js');
let ObstacleProjectedDetection = require('./ObstacleProjectedDetection.js');
let AprilTagsWithInfos = require('./AprilTagsWithInfos.js');
let AprilTagDetection = require('./AprilTagDetection.js');
let LEDPattern = require('./LEDPattern.js');
let DiagnosticsRosLink = require('./DiagnosticsRosLink.js');
let WheelsCmd = require('./WheelsCmd.js');
let WheelsCmdDBV2Stamped = require('./WheelsCmdDBV2Stamped.js');
let WheelsCmdStamped = require('./WheelsCmdStamped.js');
let Pixel = require('./Pixel.js');
let VehicleCorners = require('./VehicleCorners.js');
let IntersectionPose = require('./IntersectionPose.js');
let LightSensor = require('./LightSensor.js');
let DiagnosticsRosProfiling = require('./DiagnosticsRosProfiling.js');
let ObstacleImageDetectionList = require('./ObstacleImageDetectionList.js');
let MaintenanceState = require('./MaintenanceState.js');
let KinematicsWeights = require('./KinematicsWeights.js');
let CoordinationClearance = require('./CoordinationClearance.js');
let LEDDetection = require('./LEDDetection.js');
let DiagnosticsRosNode = require('./DiagnosticsRosNode.js');
let ThetaDotSample = require('./ThetaDotSample.js');
let StopLineReading = require('./StopLineReading.js');
let BoolStamped = require('./BoolStamped.js');
let StreetNames = require('./StreetNames.js');
let DroneMode = require('./DroneMode.js');
let DiagnosticsRosTopicArray = require('./DiagnosticsRosTopicArray.js');
let ObstacleProjectedDetectionList = require('./ObstacleProjectedDetectionList.js');
let SignalsDetectionETHZ17 = require('./SignalsDetectionETHZ17.js');
let TurnIDandType = require('./TurnIDandType.js');
let Rects = require('./Rects.js');
let DroneControl = require('./DroneControl.js');
let SourceTargetNodes = require('./SourceTargetNodes.js');
let DuckiebotLED = require('./DuckiebotLED.js');
let DiagnosticsCodeProfiling = require('./DiagnosticsCodeProfiling.js');
let Pose2DStamped = require('./Pose2DStamped.js');
let IntersectionPoseImgDebug = require('./IntersectionPoseImgDebug.js');
let EpisodeStart = require('./EpisodeStart.js');
let DiagnosticsRosParameterArray = require('./DiagnosticsRosParameterArray.js');
let LEDInterpreter = require('./LEDInterpreter.js');
let SignalsDetection = require('./SignalsDetection.js');
let CoordinationSignal = require('./CoordinationSignal.js');
let ButtonEvent = require('./ButtonEvent.js');
let DuckieSensor = require('./DuckieSensor.js');
let Segment = require('./Segment.js');
let AntiInstagramThresholds = require('./AntiInstagramThresholds.js');
let VehiclePose = require('./VehiclePose.js');
let EncoderStamped = require('./EncoderStamped.js');
let Twist2DStamped = require('./Twist2DStamped.js');
let DiagnosticsRosLinkArray = require('./DiagnosticsRosLinkArray.js');
let Rect = require('./Rect.js');
let ObstacleImageDetection = require('./ObstacleImageDetection.js');
let DisplayFragment = require('./DisplayFragment.js');
let DiagnosticsCodeProfilingArray = require('./DiagnosticsCodeProfilingArray.js');
let CarControl = require('./CarControl.js');
let Vector2D = require('./Vector2D.js');
let AprilTagDetectionArray = require('./AprilTagDetectionArray.js');
let SegmentList = require('./SegmentList.js');
let ParamTuner = require('./ParamTuner.js');
let Vsample = require('./Vsample.js');
let ObstacleType = require('./ObstacleType.js');
let LanePose = require('./LanePose.js');
let KinematicsParameters = require('./KinematicsParameters.js');
let StreetNameDetection = require('./StreetNameDetection.js');
let LineFollowerStamped = require('./LineFollowerStamped.js');
let LEDDetectionDebugInfo = require('./LEDDetectionDebugInfo.js');
let LEDDetectionArray = require('./LEDDetectionArray.js');

module.exports = {
  WheelEncoderStamped: WheelEncoderStamped,
  TagInfo: TagInfo,
  IntersectionPoseImg: IntersectionPoseImg,
  DiagnosticsRosTopic: DiagnosticsRosTopic,
  DiagnosticsRosProfilingUnit: DiagnosticsRosProfilingUnit,
  SceneSegments: SceneSegments,
  Trajectory: Trajectory,
  FSMState: FSMState,
  NodeParameter: NodeParameter,
  ObstacleProjectedDetection: ObstacleProjectedDetection,
  AprilTagsWithInfos: AprilTagsWithInfos,
  AprilTagDetection: AprilTagDetection,
  LEDPattern: LEDPattern,
  DiagnosticsRosLink: DiagnosticsRosLink,
  WheelsCmd: WheelsCmd,
  WheelsCmdDBV2Stamped: WheelsCmdDBV2Stamped,
  WheelsCmdStamped: WheelsCmdStamped,
  Pixel: Pixel,
  VehicleCorners: VehicleCorners,
  IntersectionPose: IntersectionPose,
  LightSensor: LightSensor,
  DiagnosticsRosProfiling: DiagnosticsRosProfiling,
  ObstacleImageDetectionList: ObstacleImageDetectionList,
  MaintenanceState: MaintenanceState,
  KinematicsWeights: KinematicsWeights,
  CoordinationClearance: CoordinationClearance,
  LEDDetection: LEDDetection,
  DiagnosticsRosNode: DiagnosticsRosNode,
  ThetaDotSample: ThetaDotSample,
  StopLineReading: StopLineReading,
  BoolStamped: BoolStamped,
  StreetNames: StreetNames,
  DroneMode: DroneMode,
  DiagnosticsRosTopicArray: DiagnosticsRosTopicArray,
  ObstacleProjectedDetectionList: ObstacleProjectedDetectionList,
  SignalsDetectionETHZ17: SignalsDetectionETHZ17,
  TurnIDandType: TurnIDandType,
  Rects: Rects,
  DroneControl: DroneControl,
  SourceTargetNodes: SourceTargetNodes,
  DuckiebotLED: DuckiebotLED,
  DiagnosticsCodeProfiling: DiagnosticsCodeProfiling,
  Pose2DStamped: Pose2DStamped,
  IntersectionPoseImgDebug: IntersectionPoseImgDebug,
  EpisodeStart: EpisodeStart,
  DiagnosticsRosParameterArray: DiagnosticsRosParameterArray,
  LEDInterpreter: LEDInterpreter,
  SignalsDetection: SignalsDetection,
  CoordinationSignal: CoordinationSignal,
  ButtonEvent: ButtonEvent,
  DuckieSensor: DuckieSensor,
  Segment: Segment,
  AntiInstagramThresholds: AntiInstagramThresholds,
  VehiclePose: VehiclePose,
  EncoderStamped: EncoderStamped,
  Twist2DStamped: Twist2DStamped,
  DiagnosticsRosLinkArray: DiagnosticsRosLinkArray,
  Rect: Rect,
  ObstacleImageDetection: ObstacleImageDetection,
  DisplayFragment: DisplayFragment,
  DiagnosticsCodeProfilingArray: DiagnosticsCodeProfilingArray,
  CarControl: CarControl,
  Vector2D: Vector2D,
  AprilTagDetectionArray: AprilTagDetectionArray,
  SegmentList: SegmentList,
  ParamTuner: ParamTuner,
  Vsample: Vsample,
  ObstacleType: ObstacleType,
  LanePose: LanePose,
  KinematicsParameters: KinematicsParameters,
  StreetNameDetection: StreetNameDetection,
  LineFollowerStamped: LineFollowerStamped,
  LEDDetectionDebugInfo: LEDDetectionDebugInfo,
  LEDDetectionArray: LEDDetectionArray,
};
