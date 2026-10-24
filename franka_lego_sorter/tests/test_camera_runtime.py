import os,types,unittest
from unittest.mock import Mock,patch
from lego_sorter.camera import configure_depth_units,configure_depth_options,limit_vision_threads

class CameraRuntimeTests(unittest.TestCase):
    def test_depth_preset_actual_values_recorded_and_default_read_only(self):
        sdk=types.SimpleNamespace(option=types.SimpleNamespace(visual_preset='preset',emitter_enabled='emitter',laser_power='laser'))
        values={'preset':0.,'emitter':1.,'laser':150.};sensor=Mock();sensor.supports.return_value=True
        sensor.get_option_range.return_value=types.SimpleNamespace(min=0.,max=7.,step=1.)
        sensor.get_option.side_effect=lambda option:values[option]
        sensor.set_option.side_effect=lambda option,value:values.__setitem__(option,value)
        metadata=configure_depth_options(sensor,sdk,{'depth_visual_preset':3})
        self.assertEqual(metadata['depth_visual_preset'],3);sensor.set_option.assert_called_once_with('preset',3.)
        sensor.set_option.reset_mock();configure_depth_options(sensor,sdk,{})
        sensor.set_option.assert_not_called()
    def test_depth_options_validate_all_before_setting_any(self):
        sdk=types.SimpleNamespace(option=types.SimpleNamespace(visual_preset='preset',emitter_enabled='emitter',laser_power='laser'))
        sensor=Mock();sensor.supports.return_value=True
        sensor.get_option_range.return_value=types.SimpleNamespace(min=0.,max=7.,step=1.)
        with self.assertRaises(ValueError):configure_depth_options(sensor,sdk,{'depth_visual_preset':3,'depth_laser_power':8})
        sensor.set_option.assert_not_called()
        with self.assertRaises(ValueError):configure_depth_options(sensor,sdk,{'depth_visual_preset':3.5})
        sensor.set_option.assert_not_called()
    def test_optional_units_reread_actual_sensor_scale(self):
        sensor=Mock();sensor.supports.return_value=True
        sensor.get_option_range.return_value=types.SimpleNamespace(min=.00001,max=.01)
        sensor.get_depth_scale.return_value=.0001
        sdk=types.SimpleNamespace(option=types.SimpleNamespace(depth_units='depth_units'))
        self.assertAlmostEqual(configure_depth_units(sensor,sdk,.0001),.0001)
        sensor.set_option.assert_called_once_with('depth_units',.0001)
    def test_bad_or_unapplied_units_rejected(self):
        sdk=types.SimpleNamespace(option=types.SimpleNamespace(depth_units='depth_units'))
        sensor=Mock();sensor.supports.return_value=True
        sensor.get_option_range.return_value=types.SimpleNamespace(min=.00001,max=.01)
        for value in (0,float('nan'),.1):
            with self.subTest(value=value),self.assertRaises(ValueError):configure_depth_units(sensor,sdk,value)
        sensor.get_depth_scale.return_value=.001
        with self.assertRaisesRegex(RuntimeError,'did not apply'):configure_depth_units(sensor,sdk,.0001)
    def test_default_units_never_changes_sensor_option(self):
        sensor=Mock();sensor.get_depth_scale.return_value=.001
        self.assertEqual(configure_depth_units(sensor,object()),.001);sensor.set_option.assert_not_called()
    def test_local_opencv_thread_limit_explicit(self):
        with patch.dict(os.environ,{'LEGO_ANALYSIS_THREADS':'1'}):self.assertEqual(limit_vision_threads(),1)
        with patch.dict(os.environ,{'LEGO_ANALYSIS_THREADS':'0'}),self.assertRaises(ValueError):limit_vision_threads()

if __name__=='__main__':unittest.main()
