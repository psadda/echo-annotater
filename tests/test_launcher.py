import wx


def test_launcher_runs_the_applications_main():
    import echo_annotator
    import launcher

    assert launcher.main is echo_annotator.main


def test_windows_in_tests_use_a_private_settings_file(window):
    """Guards against tests writing to the real user profile (the registry on Windows)."""
    assert isinstance(window._settings, wx.FileConfig)  # not the registry-backed wx.Config
