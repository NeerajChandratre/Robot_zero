from launch import LaunchDescription
from launch_ros.actions import Node
 
def generate_launch_description():
 
    tf2_bs_lnk_lsr_frme_node = Node(package='tf2_ros',
                    executable='static_transform_publisher',
                    name='static_tf_pub_laser',
                    arguments=['0.18', '-0.17', '0.1','0', '0', '0', '1','base_link','laser_frame'],
                    )
    tf2_bs_lnk_camra_frame_node = Node(package='tf2_ros',
                    executable='static_transform_publisher',
                    name='static_tf_pub_laser',
                    arguments=['0.08', '-0.05', '0.27','0.0333703', '0.02803', '-0.76488', '0.6426','base_link','camera_link'],
                    )
 
    return LaunchDescription([
    tf2_bs_lnk_lsr_frme_node,
    tf2_bs_lnk_camra_frame_node
    ])