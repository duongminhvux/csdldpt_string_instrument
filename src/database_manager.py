import json
from typing import Any, Dict, List, Optional

import mysql.connector
from mysql.connector import Error


class DatabaseManager:
    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
    ) -> None:
        self.config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
        }
        self.connection = None

    def connect(self) -> None:
        try:
            self.connection = mysql.connector.connect(**self.config)
            if self.connection.is_connected():
                print("Connected to MySQL successfully.")
        except Error as e:
            print(f"Error while connecting to MySQL: {e}")
            raise

    def close(self) -> None:
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("MySQL connection closed.")

    def _ensure_connection(self) -> None:
        if self.connection is None or not self.connection.is_connected():
            self.connect()

    def _execute(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetchone: bool = False,
        fetchall: bool = False,
        commit: bool = False,
    ) -> Any:
        self._ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(query, params)

            result = None
            if fetchone:
                result = cursor.fetchone()
            elif fetchall:
                result = cursor.fetchall()

            if commit:
                self.connection.commit()

            return result
        except Error as e:
            if self.connection:
                self.connection.rollback()
            print(f"Database query error: {e}")
            raise
        finally:
            cursor.close()

    def get_instrument_id(self, instrument_name: str) -> Optional[int]:
        query = """
            SELECT instrument_id
            FROM instruments
            WHERE instrument_name = %s
        """
        row = self._execute(query, (instrument_name,), fetchone=True)
        return row["instrument_id"] if row else None

    def insert_instrument(
        self,
        instrument_name: str,
        instrument_family: str = "string",
        description: Optional[str] = None,
    ) -> int:
        existing_id = self.get_instrument_id(instrument_name)
        if existing_id is not None:
            return existing_id

        query = """
            INSERT INTO instruments (instrument_name, instrument_family, description)
            VALUES (%s, %s, %s)
        """
        self._execute(
            query,
            (instrument_name, instrument_family, description),
            commit=True,
        )
        return self.get_instrument_id(instrument_name)

    def insert_audio_file(
        self,
        file_name: str,
        file_path: str,
        dataset_type: str,
        instrument_name: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        sample_rate: Optional[int] = None,
        bit_depth: Optional[int] = None,
        channels: Optional[int] = None,
        file_size_bytes: Optional[int] = None,
        file_format: str = "wav",
        notes: Optional[str] = None,
    ) -> int:
        instrument_id = None
        if instrument_name:
            instrument_id = self.insert_instrument(instrument_name)

        query = """
            INSERT INTO audio_files (
                file_name,
                file_path,
                instrument_id,
                dataset_type,
                duration_seconds,
                sample_rate,
                bit_depth,
                channels,
                file_size_bytes,
                file_format,
                notes
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            file_name,
            file_path,
            instrument_id,
            dataset_type,
            duration_seconds,
            sample_rate,
            bit_depth,
            channels,
            file_size_bytes,
            file_format,
            notes,
        )
        self._execute(query, params, commit=True)

        get_id_query = """
            SELECT audio_id
            FROM audio_files
            WHERE file_path = %s
            ORDER BY audio_id DESC
            LIMIT 1
        """
        row = self._execute(get_id_query, (file_path,), fetchone=True)
        return row["audio_id"]

    def insert_audio_features(
        self,
        audio_id: int,
        features: Dict[str, Any],
    ) -> None:
        feature_vector_json = json.dumps(features, ensure_ascii=False)

        query = """
            INSERT INTO audio_features (
                audio_id,
                rms_mean, rms_std,
                zcr_mean, zcr_std,
                attack_time,
                spectral_centroid_mean, spectral_centroid_std,
                spectral_bandwidth_mean, spectral_bandwidth_std,
                spectral_rolloff_mean, spectral_rolloff_std,
                pitch_mean, pitch_std,

                mfcc_1_mean, mfcc_2_mean, mfcc_3_mean, mfcc_4_mean, mfcc_5_mean,
                mfcc_6_mean, mfcc_7_mean, mfcc_8_mean, mfcc_9_mean, mfcc_10_mean,
                mfcc_11_mean, mfcc_12_mean, mfcc_13_mean,

                mfcc_1_std, mfcc_2_std, mfcc_3_std, mfcc_4_std, mfcc_5_std,
                mfcc_6_std, mfcc_7_std, mfcc_8_std, mfcc_9_std, mfcc_10_std,
                mfcc_11_std, mfcc_12_std, mfcc_13_std,

                chroma_1_mean, chroma_2_mean, chroma_3_mean, chroma_4_mean,
                chroma_5_mean, chroma_6_mean, chroma_7_mean, chroma_8_mean,
                chroma_9_mean, chroma_10_mean, chroma_11_mean, chroma_12_mean,

                feature_vector_json
            )
            VALUES (
                %s,
                %s, %s,
                %s, %s,
                %s,
                %s, %s,
                %s, %s,
                %s, %s,
                %s, %s,

                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s,

                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s,

                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,

                %s
            )
            ON DUPLICATE KEY UPDATE
                rms_mean = VALUES(rms_mean),
                rms_std = VALUES(rms_std),
                zcr_mean = VALUES(zcr_mean),
                zcr_std = VALUES(zcr_std),
                attack_time = VALUES(attack_time),
                spectral_centroid_mean = VALUES(spectral_centroid_mean),
                spectral_centroid_std = VALUES(spectral_centroid_std),
                spectral_bandwidth_mean = VALUES(spectral_bandwidth_mean),
                spectral_bandwidth_std = VALUES(spectral_bandwidth_std),
                spectral_rolloff_mean = VALUES(spectral_rolloff_mean),
                spectral_rolloff_std = VALUES(spectral_rolloff_std),
                pitch_mean = VALUES(pitch_mean),
                pitch_std = VALUES(pitch_std),
                mfcc_1_mean = VALUES(mfcc_1_mean),
                mfcc_2_mean = VALUES(mfcc_2_mean),
                mfcc_3_mean = VALUES(mfcc_3_mean),
                mfcc_4_mean = VALUES(mfcc_4_mean),
                mfcc_5_mean = VALUES(mfcc_5_mean),
                mfcc_6_mean = VALUES(mfcc_6_mean),
                mfcc_7_mean = VALUES(mfcc_7_mean),
                mfcc_8_mean = VALUES(mfcc_8_mean),
                mfcc_9_mean = VALUES(mfcc_9_mean),
                mfcc_10_mean = VALUES(mfcc_10_mean),
                mfcc_11_mean = VALUES(mfcc_11_mean),
                mfcc_12_mean = VALUES(mfcc_12_mean),
                mfcc_13_mean = VALUES(mfcc_13_mean),
                mfcc_1_std = VALUES(mfcc_1_std),
                mfcc_2_std = VALUES(mfcc_2_std),
                mfcc_3_std = VALUES(mfcc_3_std),
                mfcc_4_std = VALUES(mfcc_4_std),
                mfcc_5_std = VALUES(mfcc_5_std),
                mfcc_6_std = VALUES(mfcc_6_std),
                mfcc_7_std = VALUES(mfcc_7_std),
                mfcc_8_std = VALUES(mfcc_8_std),
                mfcc_9_std = VALUES(mfcc_9_std),
                mfcc_10_std = VALUES(mfcc_10_std),
                mfcc_11_std = VALUES(mfcc_11_std),
                mfcc_12_std = VALUES(mfcc_12_std),
                mfcc_13_std = VALUES(mfcc_13_std),
                chroma_1_mean = VALUES(chroma_1_mean),
                chroma_2_mean = VALUES(chroma_2_mean),
                chroma_3_mean = VALUES(chroma_3_mean),
                chroma_4_mean = VALUES(chroma_4_mean),
                chroma_5_mean = VALUES(chroma_5_mean),
                chroma_6_mean = VALUES(chroma_6_mean),
                chroma_7_mean = VALUES(chroma_7_mean),
                chroma_8_mean = VALUES(chroma_8_mean),
                chroma_9_mean = VALUES(chroma_9_mean),
                chroma_10_mean = VALUES(chroma_10_mean),
                chroma_11_mean = VALUES(chroma_11_mean),
                chroma_12_mean = VALUES(chroma_12_mean),
                feature_vector_json = VALUES(feature_vector_json)
        """

        params = (
            audio_id,
            features.get("rms_mean"),
            features.get("rms_std"),
            features.get("zcr_mean"),
            features.get("zcr_std"),
            features.get("attack_time"),
            features.get("spectral_centroid_mean"),
            features.get("spectral_centroid_std"),
            features.get("spectral_bandwidth_mean"),
            features.get("spectral_bandwidth_std"),
            features.get("spectral_rolloff_mean"),
            features.get("spectral_rolloff_std"),
            features.get("pitch_mean"),
            features.get("pitch_std"),

            features.get("mfcc_1_mean"),
            features.get("mfcc_2_mean"),
            features.get("mfcc_3_mean"),
            features.get("mfcc_4_mean"),
            features.get("mfcc_5_mean"),
            features.get("mfcc_6_mean"),
            features.get("mfcc_7_mean"),
            features.get("mfcc_8_mean"),
            features.get("mfcc_9_mean"),
            features.get("mfcc_10_mean"),
            features.get("mfcc_11_mean"),
            features.get("mfcc_12_mean"),
            features.get("mfcc_13_mean"),

            features.get("mfcc_1_std"),
            features.get("mfcc_2_std"),
            features.get("mfcc_3_std"),
            features.get("mfcc_4_std"),
            features.get("mfcc_5_std"),
            features.get("mfcc_6_std"),
            features.get("mfcc_7_std"),
            features.get("mfcc_8_std"),
            features.get("mfcc_9_std"),
            features.get("mfcc_10_std"),
            features.get("mfcc_11_std"),
            features.get("mfcc_12_std"),
            features.get("mfcc_13_std"),

            features.get("chroma_1_mean"),
            features.get("chroma_2_mean"),
            features.get("chroma_3_mean"),
            features.get("chroma_4_mean"),
            features.get("chroma_5_mean"),
            features.get("chroma_6_mean"),
            features.get("chroma_7_mean"),
            features.get("chroma_8_mean"),
            features.get("chroma_9_mean"),
            features.get("chroma_10_mean"),
            features.get("chroma_11_mean"),
            features.get("chroma_12_mean"),

            feature_vector_json,
        )

        self._execute(query, params, commit=True)

    def fetch_dataset_features(self) -> List[Dict[str, Any]]:
        query = """
            SELECT
                af.audio_id,
                af.file_name,
                af.file_path,
                i.instrument_name,
                feat.*
            FROM audio_files af
            INNER JOIN audio_features feat ON af.audio_id = feat.audio_id
            LEFT JOIN instruments i ON af.instrument_id = i.instrument_id
            WHERE af.dataset_type = 'dataset'
            ORDER BY af.audio_id
        """
        return self._execute(query, fetchall=True)

    def fetch_audio_by_id(self, audio_id: int) -> Optional[Dict[str, Any]]:
        query = """
            SELECT
                af.audio_id,
                af.file_name,
                af.file_path,
                af.dataset_type,
                af.duration_seconds,
                af.sample_rate,
                af.bit_depth,
                af.channels,
                af.file_size_bytes,
                af.file_format,
                i.instrument_name
            FROM audio_files af
            LEFT JOIN instruments i ON af.instrument_id = i.instrument_id
            WHERE af.audio_id = %s
        """
        return self._execute(query, (audio_id,), fetchone=True)

    def insert_search_query(
        self,
        query_audio_id: int,
        similarity_method: str = "cosine",
        top_k: int = 5,
        notes: Optional[str] = None,
    ) -> int:
        query = """
            INSERT INTO search_queries (
                query_audio_id,
                similarity_method,
                top_k,
                notes
            )
            VALUES (%s, %s, %s, %s)
        """
        self._execute(
            query,
            (query_audio_id, similarity_method, top_k, notes),
            commit=True,
        )

        get_id_query = """
            SELECT query_id
            FROM search_queries
            WHERE query_audio_id = %s
            ORDER BY query_id DESC
            LIMIT 1
        """
        row = self._execute(get_id_query, (query_audio_id,), fetchone=True)
        return row["query_id"]

    def insert_search_results(
        self,
        query_id: int,
        results: List[Dict[str, Any]],
    ) -> None:
        query = """
            INSERT INTO search_results (
                query_id,
                matched_audio_id,
                rank_position,
                similarity_score,
                distance_score
            )
            VALUES (%s, %s, %s, %s, %s)
        """

        self._ensure_connection()
        cursor = self.connection.cursor()
        try:
            for item in results:
                cursor.execute(
                    query,
                    (
                        query_id,
                        item["matched_audio_id"],
                        item["rank_position"],
                        item.get("similarity_score"),
                        item.get("distance_score"),
                    ),
                )
            self.connection.commit()
        except Error as e:
            self.connection.rollback()
            print(f"Error inserting search results: {e}")
            raise
        finally:
            cursor.close()