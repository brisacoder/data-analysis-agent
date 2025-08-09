#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a best-effort dictionary that maps automotive CSV signal names to human-readable
meanings, units, and domains.

- Uses exact mappings for many common signals.
- Uses regex pattern rules to cover entire signal families.
- Falls back to heuristics for anything unknown.

Outputs:
  - signal_dictionary.json
  - signal_dictionary.csv

Usage:
  python build_signal_dictionary.py

Author: ChatGPT
"""

from __future__ import annotations

import csv
import json
import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1) Paste your CSV header here (exactly as provided)
# ---------------------------------------------------------------------------

CSV_HEADER = (
    "Signal Event Time,Trip_ID,Trip_Color,Latitude,Longitude,ABS_WrngLmpSta,ADAS_ActToiSta,"
    "AWD_CltchThrmlStrssVal,AmbtPresMdl,BJD_Flag,CF_EMS_KEYST,CLU_AdasWarnSndStat,CLU_DrvngModSwSta,"
    "CLU_DtntOutSta,CatHeatBit,DATC_AUTOStep,DATC_AcDis,DATC_AcnCompOnReq,DATC_AcnReq,DATC_AlvCnt10Val,"
    "DATC_AlvCnt13Val,DATC_AlvCnt6Val,DATC_BlwModSta,DATC_BlwSta,DATC_Crc10Val,DATC_Crc13Val,DATC_Crc6Val,"
    "DATC_CtrlablSta,DATC_FnDstSnsrVal,DATC_FrChgReqDis,DATC_IntakeSta,DATC_IsgInhbtReq,DATC_MaxBlwVolt,"
    "DATC_MltAutoBlrTxt,DATC_MnlACSta,DATC_MnlIntakeSta,DATC_MnlModSta,DATC_OpSta,DATC_StrWhlHtngReq,"
    "DAW_LVDA_OptUsmSta,DBC_ClusterDis,DBC_Sta,DBC_WrngLmpSta,DCT_BrkSwSta,EBD_WrngLmpSta,EMSV_FlagSyncStat,"
    "EMSV_LockUpClthStat,EMSV_MisfStat,EMSV_NumOfMisfCyl1,EMSV_NumOfMisfCyl2,EMSV_SyncStat,ENG_Ack4EngStpSta,"
    "ENG_AirCompRlySta,ENG_AppAccelPdlSta,ENG_AtmsphPrsrVal,ENG_BattWrngLmpSta,ENG_BrkSwSta,ENG_CltchOpSta,"
    "ENG_EngSta,ENG_EtcLmphModSta,ENG_FuelCutOffSta,ENG_IgnOnSta,ENG_MaxAirconTqVal,ENG_MilSta,ENG_Neutral_MT,"
    "ENG_OilLifeRatio,ENG_OilPrsrWrngLmpSta,ENG_PwrAutoOffInfo,ENG_PwrAutoOffTimerReset,ENG_SpltInjctnSta,"
    "EPB_FrcErrSta,EPB_LmpStaDis,ESC_BcaRPlusSta,ESC_CylPrsrFlagSta,ESC_DrvBrkActvSta,ESC_DrvBrkSta,"
    "ESC_PrkBrkActvSta,ESC_RccaSta,ESC_StdStillVal,FCA_Equip_FR_CMR,FCA_FrOnOffEquipSta,FCA_Regulation,"
    "FCA_WarnTimeSta,FCA_WrngLvlSta,FCA_WrngSndSta,FCA_WrngTrgtDis,FR_CMR_FCAEquipSta,FR_CMR_MDPSCtrlSta,"
    "FR_CMR_ReqADASMapMsgVal,FR_CMR_SCCEquipSta,FR_CMR_SwVer1Val,FR_CMR_SwVer2Val,HAC_Sta,HBA_OptUsmSta,"
    "HBA_SysOptSta,HBA_SysSta,HDA_InfoPUDis,HDA_OptUsmSta,HPT_StrWhlWarn1Sta,HPT_StrWhlWarn2Sta,ISLA_SpdWrn,"
    "Info_LtLnDptSta,Info_RtLnDptSta,LKA_OnOffEquip2Sta,LKA_RcgSta,LKA_WarnSndUSMSta,LamFBCtl,LamRatioFBActivBit,"
    "MAP_DiaEnbl,MDPS_CurrModVal,MDPS_LkaPlgInSta,MDPS_LkaToiActvSta,MDPS_LkaToiUnblSta,MDPS_VsmPlgInSta,"
    "MV_DrvLnCtLnSta,MV_LtLnSta,MV_RtLnSta,OBD_RbmInhbtSta,OSMrrLamp_LtIndSta,PU_F_Group1_ADASWarn1_1Sta,"
    "PU_F_Group1_ADASWarn1_2Sta,PU_F_Group4_ADASWarn1_1Sta,PU_F_Group7_BlndSptSftyFlrSta,PU_PV_HostVehSta,"
    "PU_PV_LblSta,PU_PV_RrLtObjMvngDirSta,Purge_OnBit,RCCA_Sta,RCCW_LtIndSta,RCCW_RtSndWrngSta,RCCW_Sta,"
    "SAS_IntSta,SCC_Char2Sta,SCC_DrvOvrRidSta,SCU_FF_WrnMsgClu,SND_ADASWarn1_1Sta,SND_ADASWarn2_1Sta,"
    "TCS_LmpSta,TCS_OffLmpSta,TCU_EngRpmDisSta,TCU_FuelCutInhbtReq,TCU_GearShftSta,TCU_InertiaPhaseSta,"
    "TCU_IsgInhbtReq,TPMS_SnsrSta,TPMS_WrngLmpSta,TT_FwdSftySymbSta,TT_ISLA_SpdLimTrffcSgnSta,USM_CluAdasVolSta,"
    "ti2ndInj,ti3rdInj,ADAS_StrTqReqVal,AWD_ActrCurrVal,AWD_CltchDtVal,AWD_CltchPrsrVal,AWD_FLWhlSpdSnsrVal,"
    "AWD_FRWhlSpdSnsrVal,AWD_FrTqDistRto,AWD_RLWhlSpdSnsrVal,AWD_RRWhlSpdSnsrVal,AWD_RrTqDistRto,AWD_ShaftTqVal,"
    "AWD_ShaftTqVal_Calc,AWD_StrWhlPosVal,AccelVS,All_ClntTempGauge,All_EngSpdVal,AltDuty,CLU_AutoBrightSta,"
    "CLU_DTEVal,CLU_Dev_DTE,CLU_Dev_DTE_FCO,CLU_Dev_DTE_Fuel,CLU_Dev_DTE_Fuel_AFC_1st,CLU_Dev_DTE_Fuel_AFC_2nd,"
    "CLU_Dev_DTE_Fuel_Add,CLU_Dev_DTE_Temp,CLU_Dev_Gauge_Fuel,CLU_Dev_Reference_Fuel,CLU_Dev_Sender_Fuel,"
    "CLU_Dev_Sender_Fuel_Filtered,CLU_DisSpdDcmlVal,CLU_DisSpdVal,CLU_DisSpdVal_KPH,CLU_FuelLvlSta,"
    "CLU_NotMinModeBrightSta,CLU_OdoVal,CLU_SRSWrngLmpSta,CatDia_cntDia,CatDia_mgOSCDia,DATC_AdsActrTestVal,"
    "DATC_AdsSnsVal,DATC_AlvCnt3Val,DATC_AlvCnt5Val,DATC_AlvCnt7Val,DATC_AmbSnsVal,DATC_BlwTrgtVoltTestVal,"
    "DATC_BlwrLvl,DATC_CalcAcnCompTqVal,DATC_Crc3Val,DATC_Crc5Val,DATC_Crc7Val,DATC_DrvTempCDis,DATC_DrvTempFDis,"
    "DATC_DrvrModActrTestVal,DATC_EvaSnsVal,DATC_EvapTempSnsrCTestVal,DATC_FrDrvBlwDis,DATC_FrDrvModDis,"
    "DATC_FrIncarSnsVal,DATC_FrontBlwDuty,DATC_GlssTemp,DATC_Humidity,DATC_IncarSnsrTempCTestVal,"
    "DATC_IntakeActrTestVal,DATC_IntakeDis,DATC_LhPhotoSnsVal,DATC_MnlBlwrSta,DATC_OutTempDispC,DATC_OutTempDispF,"
    "DATC_OutTempSnsrVal,DATC_PassTempCDis,DATC_PassTempFDis,DATC_PhotoSnsrFdbckVoltTestVal,"
    "DATC_PsPhotoSnsrFdbckVoltTestVal,DATC_PsTempAct,DATC_RhPhotoSnsVal,DATC_TempActrTestVal,DAW_SysSta,"
    "DCT_EngOpSta,EMSV_ACClntPres,EMSV_ActIndTq,EMSV_AirFuelRatioBnk1,EMSV_AltDuty,EMSV_CurExCAMPos1,"
    "EMSV_CurIgAngle,EMSV_CurInCAMPos1,EMSV_KeyBattVolt,EMSV_LamCtrlOutBnk1,EMSV_LambdaOffsetDn,EMSV_MAFSetPoint,"
    "EMSV_MisfFaultCntrCat,EMSV_MisfFaultCntrEm,EMSV_ModTempClnt,EMSV_NumOfMisfCyl3,EMSV_NumOfMisfCyl4,"
    "EMSV_PurgAdapFact,EMSV_PurgAmntCani,EMSV_PurgAmntFlow,EMSV_RateInertGas,EMSV_RelAddLamAdapBnk1,"
    "EMSV_RelEngLoad,EMSV_SetExCAMPos1,EMSV_SetInCAMPos1,EMSV_SetIndTq,EMSV_ShftOptIgAngle,EMSV_TMMCurPos,"
    "EMSV_TMMStat,EMSV_TMMTarPos,EMSV_TPSActPos,EMSV_TPSSetPoint,EMSV_TarBoostPres,EMSV_TarEngTemp,EMSV_TarRPM,"
    "EMSV_TempAftCAC,ENG_AccelPdlVal,ENG_ActlAccelPdlVal,ENG_AirconPrsrSnsrVal,ENG_AltFdbckLoadVal,"
    "ENG_AmbtTempModelVal,ENG_AutoShutOffTimerMin,ENG_AutoShutOffTimerSec,ENG_BattVoltVal,ENG_BstPrsrVal,"
    "ENG_CrctEngTqVal,ENG_EngClntTempVal,ENG_EngOilTempVal,ENG_EngSpdVal,ENG_ExtSoakTimeVal,ENG_FrctTqVal,"
    "ENG_FuelCnsmptVal,ENG_IndTqBVal,ENG_IndTqVal,ENG_MafCrctVal,ENG_MaxIndTqVal,ENG_MinIndTqVal,ENG_ObdFrzFrmSta,"
    "ENG_RspaIndTqVal,ENG_SoakTimeVal,ENG_SprkTimeVal,ENG_ThrPosVal,ENG_TrgtFuelPmpPrsrVal,ENG_TrgtIdleRpmVal,"
    "ENG_TrgtTqVal,ENG_VehSpdHiVal,EPB_FlSta,EPB_FrcSta,ESC_CylPrsrVal,ESC_DrvOvrdSta,ESC_VehAccelVal,ETC_Act,"
    "ETC_SP,EngRunTime,EngSoakTime,FCA_RelVel,FuelHighPres,HDA_LFA_SymSta,ID_CIPV,ISLA_AddtnlSign,ISLA_SpdwOffst,"
    "ISLW_SpdCluMainDis,ISLW_SpdNaviMainDis,IgnDwellTime,InAirPresAct,InAirTempVal,Info_LtLnCrvtrDrvtvVal,"
    "Info_LtLnCvtrVal,Info_LtLnHdingAnglVal,Info_LtLnPosVal,Info_LtLnQualSta,Info_RtLnCrvtrDrvtvVal,Info_RtLnCvtrVal,"
    "Info_RtLnHdingAnglVal,Info_RtLnPosVal,Info_RtLnQualSta,Leak_SmllGrad,Leak_SmllstGrad,Leak_SmllstGradTh,"
    "Longitudinal_Distance,MDPS_EstStrAnglVal,MDPS_OutTqVal,MDPS_StrTqSnsrVal,MisFire_LvlTh,MisFire_Lvl_0,"
    "MisFire_Lvl_1,MisFire_Lvl_2,MisFire_Lvl_3,MisFire_NxtIgnLvlTh,MisFire_NxtIgnLvl_0,MisFire_NxtIgnLvl_1,"
    "MisFire_NxtIgnLvl_2,MisFire_NxtIgnLvl_3,OBD_AbsThrPosVal,OBD_CalcLoadVal,OBD_DnmntrCalcVal,OBD_EngClntTempVal,"
    "OBD_EngRpmVal,OBD_FuelCtrlSta,OBD_FuelRailPrsrVal,OBD_IgnCycCntVal,OBD_LongFuelTrmBnk1Val,OBD_ManAbsPrsrVal,"
    "OBD_RbmSta,OBD_ShrtFuelTrmBnk1Val,OBD_VehSpdSnsrVal,OSMrrLamp_RtIndSta,OilPres,OilTempMdl,PU_PV_OutlnColorSta,"
    "PU_PV_RrRtObjMvngDirSta,PreIgn_IntAirTemp,PreIgn_Th,Purge_LeakAirMass,RCCW_RtIndSta,Relative_Velocity,"
    "SAS_AnglVal,SAS_SpdVal,SBW_SHFTR_FF_LvrIndicatorSta,SBW_SHFTR_FF_LvrPosInfo,SCC_ObjDstVal,SCC_ObjRelSpdVal,"
    "SCU_FF_PosActSta,SCU_FF_PosSnsr,SCU_FF_PosTarSta,SMV_LFA_SymbSta,StrtClnt,TCH_EngTrgtGearSta,TCU_AtTempCVal,"
    "TCU_CluTrgtGearSta,TCU_CurrGearSta,TCU_DcmlVehSpdKphVal,TCU_EngRpmDis,TCU_EngSpdIncRev,TCU_EngTqLimVal,"
    "TCU_GearSlctDis,TCU_GrRatioChngProg,TCU_RawTqCnvrtrSpdVal,TCU_ShftPtrnSta,TCU_SlpVal,TCU_TgtTurSpd,"
    "TCU_TqCnvrtrSpdVal,TCU_TqCnvrtrSta,TCU_TqRdctnVal,TCU_TrgtGearSta,TCU_VehSpdKphVal,TMOilTemp,TPMS_FLSnsrRSSI,"
    "TPMS_FLTirePrsrVal,TPMS_FRSnsrRSSI,TPMS_FRTirePrsrVal,TPMS_NoiseRSSI,TPMS_RLSnsrRSSI,TPMS_RLTirePrsrVal,"
    "TPMS_RRSnsrRSSI,TPMS_RRTirePrsrVal,TQLrn_Igain,TQLrn_Pgain,TQLrn_TQActAtCltch,TQLrn_TQCnvtr,"
    "TQLrn_TQCompAtDrvModACOn,TQLrn_TQCompAtNeuModACOn,TT_ISLA_AddtnlTrffcSgnSta,TT_ISLA_SpdLimTrffcSgnVal,"
    "TbLrn_WstGtDuty,TnkLvl_mConsumTrip,VCRM_CanisterDutyVal,VCRM_CanisterLoadVal,VCRM_EtcMtrDutyVal,"
    "VCRM_FuelRailPrsrVal,VCRM_FuelTnkLvlVal,VCRM_FuelTnkPrsrVal,VCRM_InjctnTimeB1Val,VCRM_O2SnsrVoltB1S2Val,"
    "VCRM_TrboChrgrBstPrsrVal,VCRM_TrgtFuelRailPrsrVal,WHL_PlsFLVal,WHL_PlsFRVal,WHL_PlsRLVal,WHL_PlsRRVal,"
    "WHL_SpdFLVal,WHL_SpdFRVal,WHL_SpdRLVal,WHL_SpdRRVal,degIgnBas,degIgnRef,ti1stInj"
)

# ---------------------------------------------------------------------------
# 2) Exact mappings for common signals (extend as needed)
#    Each entry: signal -> dict(expanded=..., description=..., units=..., domain=...)
# ---------------------------------------------------------------------------

EXACT_MAP: Dict[str, Dict[str, str]] = {
    # Trip / GPS
    "Signal Event Time": {
        "expanded": "Signal Event Time",
        "description": "Timestamp of this measurement/event (likely UTC).",
        "units": "datetime",
        "domain": "Trip/Position",
    },
    "Trip_ID": {
        "expanded": "Trip Identifier",
        "description": "Unique trip/session identifier.",
        "units": "",
        "domain": "Trip/Position",
    },
    "Trip_Color": {
        "expanded": "Trip Color",
        "description": "Trip classification or UI tag color.",
        "units": "",
        "domain": "Trip/Position",
    },
    "Latitude": {
        "expanded": "Latitude",
        "description": "GPS latitude.",
        "units": "deg",
        "domain": "Trip/Position",
    },
    "Longitude": {
        "expanded": "Longitude",
        "description": "GPS longitude.",
        "units": "deg",
        "domain": "Trip/Position",
    },

    # Lamps / chassis
    "ABS_WrngLmpSta": {
        "expanded": "ABS Warning Lamp Status",
        "description": "Indicates if the ABS warning lamp is on.",
        "units": "bool",
        "domain": "Chassis/Brakes",
    },
    "EBD_WrngLmpSta": {
        "expanded": "EBD Warning Lamp Status",
        "description": "Electronic Brakeforce Distribution lamp status.",
        "units": "bool",
        "domain": "Chassis/Brakes",
    },
    "TCS_LmpSta": {
        "expanded": "TCS Lamp Status",
        "description": "Traction Control System indicator lamp status.",
        "units": "bool",
        "domain": "Chassis/Brakes",
    },
    "TCS_OffLmpSta": {
        "expanded": "TCS Off Lamp Status",
        "description": "Indicates TCS off lamp.",
        "units": "bool",
        "domain": "Chassis/Brakes",
    },

    # Engine core
    "ENG_EngSpdVal": {
        "expanded": "Engine Speed",
        "description": "Current engine crankshaft speed.",
        "units": "rpm",
        "domain": "Powertrain/Engine",
    },
    "All_EngSpdVal": {
        "expanded": "Engine Speed (Alt)",
        "description": "Engine speed (alternate source).",
        "units": "rpm",
        "domain": "Powertrain/Engine",
    },
    "OBD_EngRpmVal": {
        "expanded": "OBD Engine RPM",
        "description": "Engine RPM via OBD-II.",
        "units": "rpm",
        "domain": "Powertrain/Engine",
    },
    "ENG_EngClntTempVal": {
        "expanded": "Engine Coolant Temperature",
        "description": "Engine coolant temperature.",
        "units": "°C",
        "domain": "Powertrain/Engine",
    },
    "OBD_EngClntTempVal": {
        "expanded": "OBD Engine Coolant Temperature",
        "description": "Coolant temperature via OBD-II.",
        "units": "°C",
        "domain": "Powertrain/Engine",
    },
    "ENG_EngOilTempVal": {
        "expanded": "Engine Oil Temperature",
        "description": "Engine oil temperature.",
        "units": "°C",
        "domain": "Powertrain/Engine",
    },
    "OilTempMdl": {
        "expanded": "Oil Temperature (Model)",
        "description": "Estimated/modelled oil temperature.",
        "units": "°C",
        "domain": "Powertrain/Engine",
    },
    "ENG_BattVoltVal": {
        "expanded": "Battery Voltage",
        "description": "Electrical system battery voltage.",
        "units": "V",
        "domain": "Powertrain/Engine",
    },
    "ENG_ThrPosVal": {
        "expanded": "Throttle Position",
        "description": "Absolute throttle position.",
        "units": "%",
        "domain": "Powertrain/Engine",
    },
    "OBD_AbsThrPosVal": {
        "expanded": "OBD Absolute Throttle Position",
        "description": "Throttle position via OBD-II.",
        "units": "%",
        "domain": "Powertrain/Engine",
    },
    "ENG_BstPrsrVal": {
        "expanded": "Boost Pressure",
        "description": "Intake manifold boost pressure (post-compressor).",
        "units": "kPa",
        "domain": "Powertrain/Engine",
    },
    "FuelHighPres": {
        "expanded": "Fuel High Pressure",
        "description": "High-pressure fuel rail reading.",
        "units": "kPa",
        "domain": "Powertrain/Engine",
    },
    "VCRM_FuelRailPrsrVal": {
        "expanded": "Fuel Rail Pressure",
        "description": "Requested/measured fuel rail pressure (VCRM).",
        "units": "kPa",
        "domain": "Powertrain/Engine",
    },
    "ENG_FuelCnsmptVal": {
        "expanded": "Fuel Consumption",
        "description": "Estimated instantaneous fuel consumption.",
        "units": "L/h",
        "domain": "Powertrain/Engine",
    },
    "VCRM_FuelTnkLvlVal": {
        "expanded": "Fuel Tank Level",
        "description": "Fuel tank fill level.",
        "units": "%",
        "domain": "Powertrain/Engine",
    },
    "VCRM_FuelTnkPrsrVal": {
        "expanded": "Fuel Tank Pressure",
        "description": "Charcoal canister/tank pressure.",
        "units": "kPa",
        "domain": "Emissions/EVAP",
    },
    "ENG_AirconPrsrSnsrVal": {
        "expanded": "A/C Pressure",
        "description": "Air-conditioning refrigerant pressure.",
        "units": "kPa",
        "domain": "Body/Climate",
    },
    "ti1stInj": {
        "expanded": "Injector Pulse Width 1",
        "description": "Primary injection duration.",
        "units": "ms",
        "domain": "Powertrain/Engine",
    },
    "ti2ndInj": {
        "expanded": "Injector Pulse Width 2",
        "description": "Secondary injection duration.",
        "units": "ms",
        "domain": "Powertrain/Engine",
    },
    "ti3rdInj": {
        "expanded": "Injector Pulse Width 3",
        "description": "Tertiary injection duration.",
        "units": "ms",
        "domain": "Powertrain/Engine",
    },

    # Driveline / AWD / Wheel speeds
    "AWD_CltchPrsrVal": {
        "expanded": "AWD Clutch Pressure",
        "description": "Hydraulic pressure at AWD clutch.",
        "units": "kPa",
        "domain": "Driveline/AWD",
    },
    "AWD_FrTqDistRto": {
        "expanded": "AWD Front Torque Distribution",
        "description": "Torque share to the front axle.",
        "units": "%",
        "domain": "Driveline/AWD",
    },
    "AWD_RrTqDistRto": {
        "expanded": "AWD Rear Torque Distribution",
        "description": "Torque share to the rear axle.",
        "units": "%",
        "domain": "Driveline/AWD",
    },

    # Steering
    "MDPS_EstStrAnglVal": {
        "expanded": "Estimated Steering Angle",
        "description": "Estimated steering wheel angle.",
        "units": "deg",
        "domain": "Steering",
    },
    "MDPS_OutTqVal": {
        "expanded": "Steering Motor Output Torque",
        "description": "Assist motor output torque.",
        "units": "N·m",
        "domain": "Steering",
    },
    "SAS_AnglVal": {
        "expanded": "Steering Angle",
        "description": "Steering angle from SAS sensor.",
        "units": "deg",
        "domain": "Steering",
    },
    "SAS_SpdVal": {
        "expanded": "Steering Speed",
        "description": "Rate of steering wheel rotation.",
        "units": "deg/s",
        "domain": "Steering",
    },

    # ADAS sampling
    "ADAS_StrTqReqVal": {
        "expanded": "ADAS Steering Torque Request",
        "description": "Torque request from ADAS lane/assist.",
        "units": "N·m",
        "domain": "ADAS",
    },
    "SCC_ObjDstVal": {
        "expanded": "SCC Object Distance",
        "description": "Distance to lead object for cruise control.",
        "units": "m",
        "domain": "ADAS",
    },
    "SCC_ObjRelSpdVal": {
        "expanded": "SCC Relative Speed",
        "description": "Relative speed to lead object.",
        "units": "km/h",
        "domain": "ADAS",
    },
    "FCA_RelVel": {
        "expanded": "Forward Collision Relative Velocity",
        "description": "Relative velocity used by FCA system.",
        "units": "km/h",
        "domain": "ADAS",
    },

    # Odometer / speed / cluster
    "CLU_OdoVal": {
        "expanded": "Odometer",
        "description": "Total distance accumulated.",
        "units": "km",
        "domain": "Cluster",
    },
    "CLU_DisSpdVal_KPH": {
        "expanded": "Displayed Speed (KPH)",
        "description": "Vehicle speed shown in cluster (KPH).",
        "units": "km/h",
        "domain": "Cluster",
    },

    # TPMS lamp only (others by pattern)
    "TPMS_WrngLmpSta": {
        "expanded": "TPMS Warning Lamp Status",
        "description": "Tire pressure monitoring system lamp.",
        "units": "bool",
        "domain": "Chassis/Tires",
    },

    # Misc frequently-used
    "InAirTempVal": {
        "expanded": "Intake Air Temperature",
        "description": "Temperature of intake air.",
        "units": "°C",
        "domain": "Powertrain/Engine",
    },
    "InAirPresAct": {
        "expanded": "Intake Air Pressure",
        "description": "Actual intake manifold absolute pressure.",
        "units": "kPa",
        "domain": "Powertrain/Engine",
    },
    "OBD_VehSpdSnsrVal": {
        "expanded": "OBD Vehicle Speed",
        "description": "Vehicle speed via OBD-II.",
        "units": "km/h",
        "domain": "Powertrain/Vehicle",
    },
}


# ---------------------------------------------------------------------------
# 3) Pattern rules to decode families, e.g., WHL_*, AWD_*, TPMS_*
#    Each rule returns: (expanded, description, units, domain) or None
# ---------------------------------------------------------------------------

def rule_whl(name: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Wheel speed/pulse signals.
    """
    m_spd = re.match(r"^WHL_Spd([A-Z]{2})Val$", name)
    m_pls = re.match(r"^WHL_Pls([A-Z]{2})Val$", name)
    corner_map = {
        "FL": "Front Left",
        "FR": "Front Right",
        "RL": "Rear Left",
        "RR": "Rear Right",
    }
    if m_spd:
        corner = m_spd.group(1)
        corner_full = corner_map.get(corner, corner)
        return (f"Wheel Speed {corner_full}",
                f"Wheel speed at {corner_full} corner.",
                "km/h",
                "Chassis/Brakes")
    if m_pls:
        corner = m_pls.group(1)
        corner_full = corner_map.get(corner, corner)
        return (f"Wheel Pulses {corner_full}",
                f"Wheel pulse count at {corner_full} corner.",
                "pulses",
                "Chassis/Brakes")
    return None


def rule_awd(name: str) -> Optional[Tuple[str, str, str, str]]:
    """
    AWD wheel sensors / torque distribution / shaft torque.
    """
    if name.endswith("WhlSpdSnsrVal") and name.startswith("AWD_"):
        corner = name.split("_")[1][:2]  # FL/FR/RL/RR heuristic
        corner_map = {"FL": "Front Left", "FR": "Front Right", "RL": "Rear Left", "RR": "Rear Right"}
        cfull = corner_map.get(corner, corner)
        return (f"AWD Wheel Speed Sensor {cfull}",
                f"Wheel speed sensor reading at {cfull}.",
                "km/h",
                "Driveline/AWD")
    patterns = {
        "AWD_ShaftTqVal": ("AWD Shaft Torque", "Measured torque on AWD shaft.", "N·m", "Driveline/AWD"),
        "AWD_ShaftTqVal_Calc": ("AWD Shaft Torque (Calc)", "Calculated torque on AWD shaft.", "N·m", "Driveline/AWD"),
        "AWD_StrWhlPosVal": ("AWD Steering Wheel Position", "Steering position as seen by AWD controller.", "deg", "Driveline/AWD"),
        "AWD_ActrCurrVal": ("AWD Actuator Current", "Current to AWD actuator.", "A", "Driveline/AWD"),
        "AWD_CltchDtVal": ("AWD Clutch Duty", "Duty cycle for AWD clutch control.", "%", "Driveline/AWD"),
    }
    if name in patterns:
        return patterns[name]
    return None


def rule_tpms(name: str) -> Optional[Tuple[str, str, str, str]]:
    """
    TPMS pressures and RSSI.
    """
    m_pr = re.match(r"^TPMS_([FR][LR])TirePrsrVal$", name)
    m_rssi = re.match(r"^TPMS_([FR][LR])SnsrRSSI$", name)
    pos_map = {"FL": "Front Left", "FR": "Front Right", "RL": "Rear Left", "RR": "Rear Right"}
    if m_pr:
        pos = m_pr.group(1)
        return (f"TPMS Tire Pressure {pos_map[pos]}",
                f"Tire pressure at {pos_map[pos]}.",
                "kPa",
                "Chassis/Tires")
    if m_rssi:
        pos = m_rssi.group(1)
        return (f"TPMS Sensor RSSI {pos_map[pos]}",
                f"Signal strength from TPMS sensor at {pos_map[pos]}.",
                "dBm",
                "Chassis/Tires")
    if name == "TPMS_NoiseRSSI":
        return ("TPMS Noise RSSI",
                "Background RF noise near TPMS frequency.",
                "dBm",
                "Chassis/Tires")
    return None


def rule_obd(name: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Generic OBD-II families (fuel trims, MAP, load, etc.).
    """
    if name == "OBD_CalcLoadVal":
        return ("OBD Calculated Load", "Calculated engine load.", "%", "Powertrain/OBD")
    if name == "OBD_ManAbsPrsrVal":
        return ("OBD Manifold Absolute Pressure", "MAP via OBD-II.", "kPa", "Powertrain/OBD")
    if name == "OBD_LongFuelTrmBnk1Val":
        return ("OBD Long-Term Fuel Trim Bank 1", "LTFT Bank 1.", "%", "Powertrain/OBD")
    if name == "OBD_ShrtFuelTrmBnk1Val":
        return ("OBD Short-Term Fuel Trim Bank 1", "STFT Bank 1.", "%", "Powertrain/OBD")
    if name == "OBD_IgnCycCntVal":
        return ("OBD Ignition Cycle Count", "Number of key-on cycles.", "", "Powertrain/OBD")
    if name == "OBD_FuelRailPrsrVal":
        return ("OBD Fuel Rail Pressure", "Fuel rail pressure via OBD-II.", "kPa", "Powertrain/OBD")
    return None


def rule_tcu(name: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Transmission control unit (TCU) status.
    """
    if name == "TCU_CurrGearSta":
        return ("Current Gear", "Current selected gear from TCU.", "", "Powertrain/Transmission")
    if name == "TCU_TrgtGearSta":
        return ("Target Gear", "Target gear requested by TCU.", "", "Powertrain/Transmission")
    if name == "TCU_VehSpdKphVal":
        return ("Vehicle Speed (TCU)", "Vehicle speed from TCU.", "km/h", "Powertrain/Transmission")
    if name == "TCU_AtTempCVal":
        return ("AT Fluid Temperature", "Automatic transmission fluid temperature.", "°C", "Powertrain/Transmission")
    if name == "TCU_TqCnvrtrSpdVal":
        return ("Torque Converter Speed", "Torque converter speed.", "rpm", "Powertrain/Transmission")
    return None


def rule_datc(name: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Climate control (DATC) common signals.
    """
    if name.startswith("DATC_"):
        if "Temp" in name and name.endswith(("CDis", "SnsrVal")):
            return ("Temperature (Climate)", "Temperature value from DATC context.", "°C", "Body/Climate")
        if name in ("DATC_BlwrLvl", "DATC_BlwSta"):
            return ("Blower Level", "Blower speed/level.", "", "Body/Climate")
        if name == "DATC_AcnCompOnReq":
            return ("A/C Compressor On Request", "Request to enable A/C compressor.", "bool", "Body/Climate")
        if "Humidity" in name:
            return ("Cabin Humidity", "Cabin relative humidity.", "%", "Body/Climate")
        return ("DATC Signal", "Climate control related signal.", "", "Body/Climate")
    return None


def rule_adaptive(name: str) -> Optional[Tuple[str, str, str, str]]:
    """
    A light-touch generic rule for ADAS families (FCA, LKA, ISLA, SCC, HDA, RCCW/RCCA, etc.).
    """
    families = {
        "FCA": ("Forward Collision Assist", "ADAS/Collision"),
        "LKA": ("Lane Keeping Assist", "ADAS/Lane"),
        "ISLA": ("Intelligent Speed Limit Assist", "ADAS/Speed"),
        "SCC": ("Smart Cruise Control", "ADAS/Cruise"),
        "HDA": ("Highway Driving Assist", "ADAS/Drive"),
        "RCCW": ("Rear Cross-Traffic Collision Warning", "ADAS/Rear"),
        "RCCA": ("Rear Cross-Traffic Collision Avoidance", "ADAS/Rear"),
        "HBA": ("High Beam Assist", "ADAS/Lighting"),
        "HAC": ("Hill-Start Assist Control", "Chassis/Hill"),
    }
    prefix = name.split("_", 1)[0]
    if prefix in families:
        title, domain = families[prefix]
        return (f"{title} Signal", f"{title} related status/parameter.", "", domain)
    return None


def rule_cluster(name: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Cluster (CLU_*) signals.
    """
    if name.startswith("CLU_"):
        if "DisSpdVal_KPH" in name:
            return ("Displayed Speed (KPH)", "Speed shown by instrument cluster.", "km/h", "Cluster")
        if "OdoVal" in name:
            return ("Odometer", "Cumulative distance.", "km", "Cluster")
        if "Fuel" in name:
            return ("Cluster Fuel Metric", "Fuel-related cluster metric.", "", "Cluster")
        return ("Cluster Signal", "Instrument cluster related signal.", "", "Cluster")
    return None


def rule_misc(name: str) -> Optional[Tuple[str, str, str, str]]:
    """
    A few frequently-seen generic patterns.
    """
    if name == "AccelVS":
        return ("Vehicle Acceleration (Source)", "Vehicle longitudinal acceleration (variant).", "m/s²", "Chassis/Dynamics")
    if name == "Longitudinal_Distance":
        return ("Longitudinal Distance", "Longitudinal gap to object.", "m", "ADAS")
    if name == "IgnDwellTime":
        return ("Ignition Dwell Time", "Time coil is energized prior to spark.", "ms", "Powertrain/Ignition")
    if name == "degIgnBas":
        return ("Ignition Timing Base", "Base ignition timing.", "deg BTDC", "Powertrain/Ignition")
    if name == "degIgnRef":
        return ("Ignition Timing Reference", "Reference ignition timing.", "deg", "Powertrain/Ignition")
    return None


PATTERN_RULES = [
    rule_whl,
    rule_awd,
    rule_tpms,
    rule_obd,
    rule_tcu,
    rule_datc,
    rule_adaptive,
    rule_cluster,
    rule_misc,
]


# ---------------------------------------------------------------------------
# 4) Builder and utilities
# ---------------------------------------------------------------------------

def expand_heuristic(name: str) -> str:
    """
    Heuristic expansion: split by underscores and capitalize tokens.
    Provides a readable fallback when no exact/pattern rule matches.
    """
    tokens = re.split(r"[_]+", name)
    tokens = [t for t in tokens if t]
    return " ".join(tokens)


def classify_domain(name: str) -> str:
    """
    Fallback domain guess based on leading token.
    """
    lead = name.split("_", 1)[0]
    guess = {
        "ENG": "Powertrain/Engine",
        "OBD": "Powertrain/OBD",
        "TCU": "Powertrain/Transmission",
        "TCH": "Powertrain/Transmission",
        "AWD": "Driveline/AWD",
        "ESC": "Chassis/Brakes",
        "ABS": "Chassis/Brakes",
        "EBD": "Chassis/Brakes",
        "EPB": "Chassis/Brakes",
        "WHL": "Chassis/Brakes",
        "DATC": "Body/Climate",
        "CLU": "Cluster",
        "MDPS": "Steering",
        "SAS": "Steering",
        "FCA": "ADAS/Collision",
        "LKA": "ADAS/Lane",
        "ISLA": "ADAS/Speed",
        "SCC": "ADAS/Cruise",
        "HDA": "ADAS/Drive",
        "RCCW": "ADAS/Rear",
        "RCCA": "ADAS/Rear",
        "TPMS": "Chassis/Tires",
    }
    return guess.get(lead, "Unknown")


def apply_rules(name: str) -> Optional[Dict[str, str]]:
    """
    Try each pattern rule and convert to a standardized dict if matched.
    """
    for rule in PATTERN_RULES:
        res = rule(name)
        if res:
            expanded, description, units, domain = res
            return {
                "expanded": expanded,
                "description": description,
                "units": units,
                "domain": domain,
            }
    return None


def build_signal_dictionary(header: str) -> Dict[str, Dict[str, str]]:
    """
    Build the dictionary for all signals in the given CSV header.

    Parameters
    ----------
    header : str
        Comma-separated column names from the CSV.

    Returns
    -------
    Dict[str, Dict[str, str]]
        Mapping: signal_name -> info dict with keys:
        expanded, description, units, domain
    """
    names = [h.strip() for h in header.split(",") if h.strip()]
    mapping: Dict[str, Dict[str, str]] = {}

    for name in names:
        # 1) Exact table
        info = EXACT_MAP.get(name)
        if info:
            mapping[name] = info
            continue

        # 2) Pattern-based families
        info = apply_rules(name)
        if info:
            mapping[name] = info
            continue

        # 3) Fallback heuristic
        mapping[name] = {
            "expanded": expand_heuristic(name),
            "description": "Signal not in dictionary; description inferred heuristically.",
            "units": "",
            "domain": classify_domain(name),
        }

    return mapping


def save_as_json(mapping: Dict[str, Dict[str, str]], path: str) -> None:
    """
    Save mapping as pretty-printed JSON.

    Parameters
    ----------
    mapping : Dict[str, Dict[str, str]]
        The signal dictionary to save.
    path : str
        Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def save_as_csv(mapping: Dict[str, Dict[str, str]], path: str) -> None:
    """
    Save mapping as CSV with columns: Signal, Expanded, Description, Units, Domain.

    Parameters
    ----------
    mapping : Dict[str, Dict[str, str]]
        The signal dictionary to save.
    path : str
        Output file path.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Signal", "Expanded", "Description", "Units", "Domain"])
        for key, info in mapping.items():
            writer.writerow([
                key,
                info.get("expanded", ""),
                info.get("description", ""),
                info.get("units", ""),
                info.get("domain", ""),
            ])


def main() -> None:
    """
    Build the dictionary from the embedded header and save it to disk.
    """
    mapping = build_signal_dictionary(CSV_HEADER)
    save_as_json(mapping, "signal_dictionary.json")
    save_as_csv(mapping, "signal_dictionary.csv")

    # Tiny preview for sanity
    preview_keys = [
        "ENG_EngSpdVal",
        "OBD_ManAbsPrsrVal",
        "WHL_SpdFLVal",
        "AWD_RrTqDistRto",
        "TPMS_FLTirePrsrVal",
        "DATC_AcnCompOnReq",
        "TCU_CurrGearSta",
        "SCC_ObjDstVal",
        "CLU_OdoVal",
        "UnknownButTotallyReal_Signal123",
    ]
    print("Preview:")
    for k in preview_keys:
        info = mapping.get(k, {"expanded": "(not found in this header)"})
        print(f"  {k:30s} -> {info['expanded']}")


if __name__ == "__main__":
    main()
