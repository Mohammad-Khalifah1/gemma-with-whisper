# Block ROS2 pytest plugins that conflict with this project's pytest version.
_ROS_PLUGINS = [
    "launch_testing_ros_pytest_entrypoint",
    "launch_testing",
    "ament_copyright",
    "ament_pep257",
    "ament_flake8",
    "ament_lint",
    "ament_xmllint",
]


def pytest_configure(config):
    for plugin in _ROS_PLUGINS:
        config.pluginmanager.set_blocked(plugin)
