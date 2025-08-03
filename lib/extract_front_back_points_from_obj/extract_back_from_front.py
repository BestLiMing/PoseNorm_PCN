# ==================== CODE WATERMARK ========================
# ⚠️ This source file is authored by Ming Li.
# Unauthorized reproduction, distribution, or modification
# without explicit permission is strictly prohibited.
# ============================================================

import psbody.mesh
import os
from os.path import join, split, exists, basename, dirname, splitdrive, normpath, getsize
import pickle as pkl
import numpy as np
import tqdm
import trimesh
from pathlib import Path
import open3d as o3d
from psbody.mesh import Mesh, MeshViewer
import shutil


class FrontToBackProcess:
    def __init__(self, down_point_num: int = 20000, ray_offset: float = 0.005, min_distance: float = 0.003,
                 rays_direction: list = (0, 0, -1), show: bool = False, save: bool = False):
        self.smpl_parts_dense_path = join(dirname(__file__), "../../assets/smpl_parts_dense.pkl")
        if not exists(self.smpl_parts_dense_path):
            raise "SMPL pre-segmentation file does not exist"
        self.parts_dense = pkl.load(open(self.smpl_parts_dense_path, 'rb'), encoding='latin-1')  # load parts dense
        self.smpl_v_label = np.zeros(
            (len(np.concatenate(list(self.parts_dense.values()))), 1))  # Labeling of initialization points
        for n, k in enumerate(self.parts_dense):
            self.smpl_v_label[self.parts_dense[k]] = n  # Instantiated labels

        self.down_point_num = down_point_num
        self.ray_offset = ray_offset
        self.min_distance = min_distance
        self.rays_direction = np.array(list(rays_direction))  # ray direction
        self.show = show
        self.save = save

    @staticmethod
    def _extract_nearest_vertices(mesh: psbody.mesh.mesh.Mesh, single_points: np.ndarray):
        '''single scanning point with the nearest vertex of the SMPL mode'''
        ind, _ = mesh.closest_vertices(single_points)  # get nearest vertex index
        ind = np.asarray(ind)
        invalid_indices = np.where((ind < 0) | (ind > (mesh.v.shape[0] - 1)))[0]  # Validating Index Validity
        if len(invalid_indices) > 0:
            print(f"Number of invalid indexes found: {len(invalid_indices)}")
            valid_mask = ~np.isin(np.arange(len(ind)), invalid_indices)
            ind = ind[valid_mask]
            print(f"Remaining after deleting invalid indexes: {len(ind)}")
        mesh_closest_vertices = mesh.v[ind]  # mesh nearest scan points
        mesh_closest_faces = mesh.f[ind]  # mesh nearest scan faces
        return mesh_closest_faces, mesh_closest_vertices, ind

    def _batch_ray_cast(self, tri_mesh: trimesh.Trimesh, points: np.ndarray):
        '''ray detection'''
        ray_origins = points - np.array([0, 0, self.ray_offset])
        ray_directions = np.tile(self.rays_direction / np.linalg.norm(self.rays_direction), (len(points), 1))
        locations, ray_ids, _ = tri_mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )
        return self._process_intersections(points=points, locations=locations, ray_ids=ray_ids)

    def _process_intersections(self, points, locations, ray_ids):
        '''process intersection'''
        processed_points = np.copy(points)
        valid_mask = np.zeros(len(points), dtype=bool)

        if len(locations) > 0:
            ray_id_to_hits = {}
            for loc, rid in zip(locations, ray_ids):
                if rid not in ray_id_to_hits:
                    ray_id_to_hits[rid] = []
                ray_id_to_hits[rid].append(loc)

            for rid, hits in ray_id_to_hits.items():
                distances = [np.linalg.norm(hit - points[rid]) for hit in hits]
                valid_distances = [d for d in distances if d > self.min_distance]

                if len(valid_distances) > 0:
                    max_idx = np.argmax(valid_distances)
                    processed_points[rid] = hits[max_idx]
                    valid_mask[rid] = True
        return processed_points, valid_mask

    @staticmethod
    def _mesh_to_trimesh(psbody_mesh: psbody.mesh.mesh.Mesh) -> trimesh.Trimesh:
        assert hasattr(psbody_mesh, 'v') and hasattr(psbody_mesh, 'f'), "无效的网格对象"
        return trimesh.Trimesh(vertices=psbody_mesh.v, faces=psbody_mesh.f, process=False)

    def _extract_shelter_points(self, mesh: psbody.mesh.mesh.Mesh, single_points: np.ndarray,
                                single_labels: np.ndarray = None):
        '''shelter point extraction'''
        tri_mesh = self._mesh_to_trimesh(mesh)  # convert mesh format
        processed_points, valid_mask = self._batch_ray_cast(tri_mesh=tri_mesh,
                                                            points=single_points)  # Batch radiography
        if single_labels is not None:
            shelter_labels = single_labels.copy()  # valid label
        else:
            shelter_labels = single_labels
        return processed_points, shelter_labels, valid_mask

    def process_file_t(self, t_scan, front_points: np.ndarray):
        ind, _ = t_scan.closest_vertices(front_points)

        _, scan_body_mapping_points = t_scan.closest_faces_and_points(front_points)
        _, _, ind = self._extract_nearest_vertices(mesh=t_scan, single_points=front_points)
        scan_body_mapping_points_labels = self.smpl_v_label[ind]
        scan_body_shelter_points, _, _ = self._extract_shelter_points(
            mesh=t_scan,
            single_points=scan_body_mapping_points,
            single_labels=scan_body_mapping_points_labels)
        assert len(scan_body_shelter_points) == len(
            scan_body_mapping_points), "Inconsistent number of scan points output"
        return scan_body_shelter_points

    def process_file(self, path):
        name = basename(path)

        # get files
        param_path = join(path, "parameters.pkl")
        scan_body_path = join(path, "scaled", name + "_scaled.obj")
        body_path = join(path, "scaled", name + "_scaled_naked.obj")
        # front_point_path = join(path, "front_views", "front.ply")

        # get front point path from other path
        # other save path
        # path_parts = normpath(path).split(os.sep)
        # valid_path = join(*path_parts[2:])
        # extract_root = Path("F:" + valid_path + "/front_views")

        # current input path
        extract_root = Path(path + "/front_views")
        os.makedirs(extract_root, exist_ok=True)
        front_point_path = join(extract_root, "front.ply")

        # front_point_path = r"F:\cape_data\00032\longshort_ATUsquat\longshort_ATUsquat.000001\front_views\front.ply"
        # check files
        param_exist = True if exists(param_path) else False
        scan_exist = True if exists(scan_body_path) else False
        body_exist = True if exists(body_path) else False
        front_exist = True if exists(front_point_path) else False

        if front_exist is False:
            print(f"Current data {name} is missing relevant documents, delete {path} and {dirname(extract_root)}")
            # shutil.rmtree(dirname(extract_root))
            # shutil.rmtree(path)
            return False

        # check out path
        scan_body_extract_information_path = rf"{extract_root}\outside.npz"
        body_extract_information_path = rf"{extract_root}\inside.npz"

        if exists(scan_body_extract_information_path) and exists(body_extract_information_path):
            print(
                f"Extraction point files already exists. {scan_body_extract_information_path} and {body_extract_information_path}")
            return True
        else:
            if exists(scan_body_extract_information_path):
                os.remove(scan_body_extract_information_path)
            if exists(body_extract_information_path):
                os.remove(body_extract_information_path)

        # load models
        if param_exist:
            param = pkl.load(open(param_path, 'rb'),
                             encoding='latin-1')  # load parameters, keys:pose, betas, trans, gender
            gender = param['gender']  # get gender
        if scan_exist:
            scan_body = Mesh(filename=scan_body_path)  # load scan body models with clothing
        if body_exist:
            body = Mesh(filename=body_path)  # load scan body models

        front_scan = Mesh(filename=front_point_path)  # load front points

        # get models
        actual_sample_num = min(len(front_scan.v), self.down_point_num)
        rand_indices = np.random.choice(len(front_scan.v),
                                        size=actual_sample_num,
                                        replace=False)  # front points down sampling
        front_scan_sampled = np.asarray(front_scan.v[rand_indices])  # front points after down sampling

        # Real scanning points mapped on the scanned body's mesh
        if scan_exist:
            scan_body_mapping_faces, scan_body_mapping_points = scan_body.closest_faces_and_points(front_scan_sampled)
            _, _, ind = self._extract_nearest_vertices(mesh=scan_body, single_points=front_scan_sampled)
            scan_body_mapping_points_labels = self.smpl_v_label[ind]  # Labeling of mapping points
            # scan_body_mapping_points_labels = None

        # Real scanning points mapped on the body's mesh
        if body_exist:
            body_mapping_faces, body_mapping_points = body.closest_faces_and_points(front_scan_sampled)
            _, _, ind = self._extract_nearest_vertices(mesh=body, single_points=front_scan_sampled)
            body_mapping_points_labels = self.smpl_v_label[ind]  # Labeling of mapping points

        # Extract the occlusion point corresponding to the scanning point and the scanning body point
        try:
            # Extract scan shelter points and labels
            if scan_exist:
                scan_body_shelter_points, scan_body_shelter_labels, scan_valid_mask = self._extract_shelter_points(
                    mesh=scan_body,
                    single_points=scan_body_mapping_points,
                    single_labels=scan_body_mapping_points_labels)

                scan_invalid_idx = np.where(scan_valid_mask == False)
                scan_invalid_points = scan_body_shelter_points[scan_invalid_idx]
                assert len(scan_body_shelter_points) == len(
                    scan_body_mapping_points), "The number of scan points is inconsistent"

            # Extract body shelter points and labels
            if body_exist:
                body_shelter_points, body_shelter_labels, body_valid_mask = self._extract_shelter_points(
                    mesh=body,
                    single_points=body_mapping_points,
                    single_labels=body_mapping_points_labels)

                body_invalid_idx = np.where(body_valid_mask == False)
                body_invalid_points = body_shelter_points[body_invalid_idx]
                assert len(body_shelter_points) == len(body_mapping_points), "The naked output points are inconsistent"

            # print(
            #     f"无效点统计 - Scan Body: {len(scan_body_mapping_points) - scan_valid_mask.sum()}, Body: {len(body_mapping_points) - body_valid_mask.sum()}, \n"
            #     f"有效点比例统计 - Scan body: {scan_valid_mask.mean():.2%}, Body: {body_valid_mask.mean():.2%} \n"
            # )

            # self._save_results(
            #     out_path=rf"H:\W-Paper-1in3-2025-IHR\Net\dataset\FrontToBackData2\{basename(dirname(dirname(path)))}_{name}.npz",
            #     front_points=scan_body_mapping_points,
            #     back_points=scan_body_shelter_points,
            #     labels=scan_body_mapping_points_labels.squeeze().astype(int),
            #     back_mask=scan_valid_mask
            # )

            if self.save:
                if scan_exist:
                    self._save_results(out_path=scan_body_extract_information_path,
                                       front_points=scan_body_mapping_points,
                                       back_points=scan_body_shelter_points,
                                       labels=None if scan_body_mapping_points_labels is None else scan_body_mapping_points_labels.squeeze().astype(
                                           int),
                                       back_mask=scan_valid_mask)

                if body_exist:
                    self._save_results(out_path=body_extract_information_path,
                                       front_points=body_mapping_points,
                                       back_points=body_shelter_points,
                                       labels=None if body_mapping_points_labels is None else body_mapping_points_labels.squeeze().astype(
                                           int),
                                       back_mask=body_valid_mask)
            if self.show:
                self._visualize_results(
                    verts=scan_body.v,
                    faces=scan_body.f,
                    scan_points=scan_body_mapping_points,
                    shelter_points=scan_body_shelter_points,
                    # shelter_points=scan_invalid_points
                )
                self._visualize_results(
                    verts=body.v,
                    faces=body.f,
                    scan_points=body_mapping_points,
                    shelter_points=body_shelter_points,
                    # shelter_points=body_invalid_points
                )
            return True
        except Exception as e:
            print(f"Error in processing: {e}")
            raise

    def _save_results(self, out_path: str, front_points: np.ndarray, back_points: np.ndarray, labels: np.ndarray,
                      back_mask: np.ndarray):
        '''save results'''
        try:
            np.savez(
                out_path,
                count=self.down_point_num,
                front_points=front_points,
                back_points=back_points,
                labels=labels,
                back_mask=back_mask,
            )
        except Exception as e:
            print(f"save error: {e}")
            raise

    def _visualize_results(self, verts, faces, scan_points=None, shelter_points=None, invalid_points=None):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        vis.add_geometry(mesh)

        if scan_points is not None:
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(scan_points)
            pcd1.paint_uniform_color([1, 0, 0])
            vis.add_geometry(pcd1)

        if shelter_points is not None:
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(shelter_points)
            pcd2.paint_uniform_color([0, 1, 0])
            vis.add_geometry(pcd2)

        if invalid_points is not None:
            pcd3 = o3d.geometry.PointCloud()
            pcd3.points = o3d.utility.Vector3dVector(invalid_points)
            pcd3.paint_uniform_color([0, 0, 1])
            vis.add_geometry(pcd3)

        opt = vis.get_render_option()
        opt.point_size = 3.0
        opt.background_color = np.asarray([0.1, 0.1, 0.1])

        vis.run()
        vis.destroy_window()


def check_error_file(dataset_root, save_path=None):
    '''checking the availability of the front point cloud '''
    if not exists(dataset_root):
        print(f"Root does not exist. dataset: {dataset_root}")
        raise
    all_folders = []
    objects = [join(dataset_root, obj) for obj in os.listdir(dataset_root)]
    for obj in objects:
        sub_objects = [join(obj, sub_obj) for sub_obj in os.listdir(obj)]
        for sub_obj in sub_objects:
            sub_sub_objects = [join(sub_obj, sub_sub_obj) for sub_sub_obj in os.listdir(sub_obj)]
            all_folders.extend(sub_sub_objects)

    valid_all_folders = []
    for folder in tqdm.tqdm(all_folders, total=len(all_folders)):
        name = basename(folder)

        path_parts = normpath(folder).split(os.sep)
        valid_path = join(*path_parts[1:])
        scaled_root = join("H:\W-Paper-1in3-2025-IHR", valid_path, "scaled")

        front_path = join(folder, "front_views", "front.ply")
        scan_obj = join(scaled_root, f"{name}_scaled.obj")
        naked_obj = join(scaled_root, f"{name}_scaled_naked.obj")
        # print(front_path, scan_obj, naked_obj)
        if exists(front_path) and exists(scan_obj) and exists(naked_obj) and getsize(scan_obj) > 0 and getsize(
                naked_obj) > 0:
            valid_all_folders.append(folder)
        else:
            try:
                shutil.rmtree(folder)
                shutil.rmtree(dirname(scaled_root))
                print(f"Delete invalid folder. {folder} and {dirname(scaled_root)}")
            except Exception as e:
                print(f"Delete error: {e}")
                continue

    if save_path is not None:
        out_path = join(save_path, "front_views.npz")
        if exists(out_path):
            os.remove(out_path)
        np.savez(out_path, folders=valid_all_folders)


if __name__ == "__main__":
    # check_error_file(dataset_root=r"F:\cape_data", save_path=r"H:\W-Paper-1in3-2025-IHR\Net\preprocess\CAPE")

    # CAPE_all_folders = r"H:\W-Paper-1in3-2025-IHR\Net\dataset_preprocess\CAPE\CAPE_all_folders.npz"
    # if not exists(CAPE_all_folders):
    #     raise FileNotFoundError(f"数据集信息文件不存在: {CAPE_all_folders}")

    # 执行处理
    # data = np.load(CAPE_all_folders)
    # dataset_path = data["data_path"]
    # dataset_count = data["total_count"]
    # dataset_all_folders = data["all_folders"]

    # dataset_all_folders = np.load(r"H:\W-Paper-1in3-2025-IHR\Net\dataset\best_data.npz")['folders']
    # dataset_count = len(dataset_all_folders)
    #
    # processor = FrontToBackProcess()
    # success = 0
    # for folder in tqdm.tqdm(dataset_all_folders, total=dataset_count):
    #     # path = f"{dataset_path}/{str(folder)}"
    #     path = folder
    #     rs = processor.process_file(path)
    #     if rs:
    #         success += 1
    # print(f"success: {success}, fail: {dataset_count - success}")

    path = r"H:\W-Paper-1in3-2025-IHR\cape_data\00032\longshort_pose_model\longshort_pose_model.000025"
    process = FrontToBackProcess().process_file(path)
