CDF       
      nMesh2_face       Three         nMesh2_node       nMesh2_boundary    	   Two             Conventions       	UGRID-0.9      Title         UGRID for GNOME    Institution       USA/DOC/NOAA/NOS/ERD/TSSB      
References        brian.zelenke@noaa.gov           Mesh2             	   cf_role       mesh_topology      	long_name         %Topology data of 2D unstructured mesh      topology_dimension              node_coordinates      Mesh2_node_x Mesh2_node_y      face_node_connectivity        Mesh2_face_nodes   edge_node_connectivity        Mesh2_edge_nodes   face_face_connectivity        Mesh2_face_links   face_coordinates      Mesh2_face_x Mesh2_face_y      boundary_node_connectivity        Mesh2_boundary_nodes        �   Mesh2_face_nodes                   cf_role       face_node_connectivity     	long_name         5Maps every triangular face to its three corner nodes.      start_index       ?�            �  �   Mesh2_face_links                   cf_role       face_face_connectivity     	long_name         /Indicates which other faces neighbor each face.    start_index       ?�         flag_values       ��         flag_meanings         out_of_mesh       �  8   Mesh2_node_x               standard_name         	longitude      	long_name         Longitude of 2D mesh nodes.    units         degrees_east      X  �   Mesh2_node_y               standard_name         latitude   	long_name         Latitude of 2D mesh nodes.     units         degrees_north         X  ,   Mesh2_face_x                standard_name         	longitude      	long_name         HCharacteristic longitude of 2D mesh triangle (i.e. centroid coordinate).   units         degrees_east      h  �   Mesh2_face_y                standard_name         latitude   	long_name         GCharacteristic latitude of 2D mesh triangle (i.e. centroid coordinate).    units         degrees_north         h  �   Mesh2_face_u                standard_name         eastward_sea_water_velocity    units         m/s    coordinates       Mesh2_face_x Mesh2_face_y      face_methods      
area: mean        h  T   Mesh2_face_v                standard_name         northward_sea_water_velocity   units         m/s    coordinates       Mesh2_face_x Mesh2_face_y      face_methods      
area: mean        h  �   Mesh2_depth                standard_name         sea_floor_depth_below_geoid    units         m      positive      down   mesh      Mesh2      location      node   coordinates       Mesh2_node_x Mesh2_node_y         X  $   Mesh2_boundary_nodes                  cf_role       boundary_node_connectivity     	long_name         CMaps every edge of each boundary to the two nodes that it connects.    start_index       ?�            H  |   Mesh2_boundary_count               mesh      Mesh2      	long_name         ?Defines the group to which every edge of each boundary belongs.    location      boundary   start_index       ?�            $  �   Mesh2_boundary_types               	long_name         4Classification flag for every edge of each boundary.   location      boundary   mesh      Mesh2      
flag_range                ?�         flag_values               ?�         flag_meanings         closed_boundary open_boundary         $  �           
   
            
   
   
   
               	                                                               	         ����������������            ��������   ����            
      ����         	         ����   	                     
      	      �O��"���<�p���<�p���G��~�1U�J� �8*�E�/��w�B����r�C��x����F���Ii7:��O�At�Y5�;@)�̫>݋@)�̫>݋@>��Q�@?�$�/�@BT�Y�:@<ffU�n�@9r,��x@1��ҍ .@649XbN@43���@=K5�I�P�B�_`qO�E�މ�H�G&ȱm�%�IU����?��_q8�@�ߝBY�A���'�7�G��F�5c�Iz^/��	�K�������Ft�{C��E�P�N��CW���~�@>�(w�N'@-��2��@7�2y@�]@=��o�i�@6����"J@<v+-c��@2����@:��PJ��@0���j/I@76��0�@4!G��+@=ҕ�]y�@;��a�ʎ?�ȝ'Óh����s恪�ϊ�ew�?��T�*��x���W�ţ!h/���恩�ˎ?�{V�syt��ݓ�m��?p���*\?��vh<)|?���a5��'��O	S?�v�A��%�ȿИ�[�vh��B�ծ��m�?>��n��8z�ڗ1	�G|?����]�?������?���-���S;wG?��#l�?�9!����?�      ?�      ?�      @Y�     ?�      ?�      @N      ?�      ?�      @X@     ?�                                 	                        	                                                                 