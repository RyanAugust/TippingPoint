import pytest
from unittest.mock import patch, MagicMock
import sys
from tippingpoint.cli import main, launch_dashboard

class TestCLI:
    @patch('sys.argv', ['tipp', 'dashboard'])
    @patch('tippingpoint.cli.launch_dashboard')
    def test_main_dashboard(self, mock_launch):
        main()
        mock_launch.assert_called_once()

    @patch('sys.argv', ['tipp'])
    @patch('argparse.ArgumentParser.print_help')
    def test_main_help(self, mock_print_help):
        main()
        mock_print_help.assert_called_once()

    @patch('subprocess.run')
    def test_launch_dashboard_success(self, mock_run):
        launch_dashboard()
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "streamlit"
        assert args[1] == "run"
        assert "dashboard.py" in args[2]

    @patch('sys.exit', side_effect=SystemExit)
    def test_launch_dashboard_import_error(self, mock_exit):
        with patch.dict('sys.modules', {'streamlit': None}):
            with pytest.raises(SystemExit):
                launch_dashboard()
            mock_exit.assert_called_once_with(1)
