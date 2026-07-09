from setuptools import setup
import os
from glob import glob
package_name = 'yahboomcar_astra'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name,'launch'),glob(os.path.join('launch','*launch.py'))),
        (os.path.join('share',package_name,'config'),glob(os.path.join('config','*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nx-ros2',
    maintainer_email='13377528435@sina.cn',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'colorHSV = yahboomcar_astra.colorHSV:main',
        'colorTracker = yahboomcar_astra.colorTracker:main',
        'qrTracker = yahboomcar_astra.qrTracker:main',
        'mono_Tracker = yahboomcar_astra.mono_Tracker:main',
        'face_fllow = yahboomcar_astra.face_fllow:main',
        'face_fllow_tpu = yahboomcar_astra.face_fllow_tpu:main',
        'objTracker_tpu = yahboomcar_astra.objTracker_tpu:main',
        'objTracker_reid_tpu = yahboomcar_astra.objTracker_reid_tpu:main',
        'objControl = yahboomcar_astra.objControl:main',
        'person_goal_bridge = yahboomcar_astra.person_goal_bridge:main',
        'follow_line = yahboomcar_astra.follow_line:main'
        ],
    },
)
