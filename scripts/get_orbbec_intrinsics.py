from pyorbbecsdk import *

def main():
    pipeline = Pipeline()
    config = Config()
    
    try:
        # Get depth profiles
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_default_video_stream_profile()
        
        # We need to start the pipeline to get valid intrinsics usually
        config.enable_stream(depth_profile)
        pipeline.start(config)
        
        # Read the intrinsics from the device
        intrinsics = depth_profile.get_intrinsic()
        
        print("\n=== Orbbec Depth Camera Intrinsics ===")
        print(f"Resolution : {depth_profile.get_width()}x{depth_profile.get_height()}")
        print(f"fx         : {intrinsics.fx:.4f}")
        print(f"fy         : {intrinsics.fy:.4f}")
        print(f"cx         : {intrinsics.cx:.4f}")
        print(f"cy         : {intrinsics.cy:.4f}")
        print("======================================\n")
        
        print("Usage in kinova_cloth_inference.py:")
        print(f"--fx {intrinsics.fx:.1f} --fy {intrinsics.fy:.1f} --cx {intrinsics.cx:.1f} --cy {intrinsics.cy:.1f}")
        
    except Exception as e:
        print(f"Error reading intrinsics: {e}")
        print("\nMake sure the camera is plugged in and no other scripts are using it.")
    finally:
        try:
            pipeline.stop()
        except:
            pass

if __name__ == "__main__":
    main()
