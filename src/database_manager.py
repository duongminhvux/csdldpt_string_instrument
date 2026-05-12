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
        self.connection = mysql.connector.connect(**self.config)
        if self.connection.is_connected():
            print("Connected to MySQL successfully.")

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

    def clear_dataset_data(self) -> None:
        self._ensure_connection()
        cursor = self.connection.cursor()

        try:
            cursor.execute(
                """
                DELETE feat
                FROM audio_features feat
                INNER JOIN audio_files af ON feat.audio_id = af.audio_id
                WHERE af.dataset_type = 'dataset'
                """
            )

            cursor.execute(
                """
                DELETE FROM audio_files
                WHERE dataset_type = 'dataset'
                """
            )

            self.connection.commit()
            print("Old dataset data cleared successfully.")

        except Error as e:
            self.connection.rollback()
            print(f"Error clearing dataset data: {e}")
            raise

        finally:
            cursor.close()

    def reset_auto_increment_for_clean_build(self) -> None:
        self._ensure_connection()
        cursor = self.connection.cursor()

        try:
            cursor.execute("ALTER TABLE audio_features AUTO_INCREMENT = 1")
            cursor.execute("ALTER TABLE audio_files AUTO_INCREMENT = 1")
            self.connection.commit()
            print("AUTO_INCREMENT values reset successfully.")

        except Error as e:
            self.connection.rollback()
            print(f"Error resetting AUTO_INCREMENT: {e}")
            raise

        finally:
            cursor.close()

    def insert_audio_file(
        self,
        file_name: str,
        file_path: str,
        dataset_type: str = "dataset",
        duration_seconds: Optional[float] = None,
        sample_rate: Optional[int] = None,
        bit_depth: Optional[int] = None,
        channels: Optional[int] = None,
        file_size_bytes: Optional[int] = None,
        file_format: str = "wav",
        notes: Optional[str] = None,
    ) -> int:
        query = """
            INSERT INTO audio_files (
                file_name,
                file_path,
                dataset_type,
                duration_seconds,
                sample_rate,
                bit_depth,
                channels,
                file_size_bytes,
                file_format,
                notes
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        params = (
            file_name,
            file_path,
            dataset_type,
            duration_seconds,
            sample_rate,
            bit_depth,
            channels,
            file_size_bytes,
            file_format,
            notes,
        )

        self._ensure_connection()
        cursor = self.connection.cursor()

        try:
            cursor.execute(query, params)
            self.connection.commit()
            return int(cursor.lastrowid)

        except Error as e:
            self.connection.rollback()
            print(f"Error inserting audio file: {e}")
            raise

        finally:
            cursor.close()

    def insert_audio_features(
        self,
        audio_id: int,
        frame_features: List[Dict[str, Any]],
    ) -> None:
        query = """
            INSERT INTO audio_features (
                audio_id,
                frame_index,
                start_time,
                end_time,
                rms_energy,
                zero_crossing_rate,
                average_frequency,
                frequency_variation,
                average_pitch,
                pitch_variation,
                feature_vector_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                start_time = VALUES(start_time),
                end_time = VALUES(end_time),
                rms_energy = VALUES(rms_energy),
                zero_crossing_rate = VALUES(zero_crossing_rate),
                average_frequency = VALUES(average_frequency),
                frequency_variation = VALUES(frequency_variation),
                average_pitch = VALUES(average_pitch),
                pitch_variation = VALUES(pitch_variation),
                feature_vector_json = VALUES(feature_vector_json)
        """

        self._ensure_connection()
        cursor = self.connection.cursor()

        try:
            for item in frame_features:
                vector = {
                    "rms_energy": float(item.get("rms_energy", 0.0)),
                    "zero_crossing_rate": float(item.get("zero_crossing_rate", 0.0)),
                    "average_frequency": float(item.get("average_frequency", 0.0)),
                    "frequency_variation": float(item.get("frequency_variation", 0.0)),
                    "average_pitch": float(item.get("average_pitch", 0.0)),
                    "pitch_variation": float(item.get("pitch_variation", 0.0)),
                }

                params = (
                    audio_id,
                    int(item.get("frame_index", 0)),
                    float(item.get("start_time", 0.0)),
                    float(item.get("end_time", 0.0)),
                    vector["rms_energy"],
                    vector["zero_crossing_rate"],
                    vector["average_frequency"],
                    vector["frequency_variation"],
                    vector["average_pitch"],
                    vector["pitch_variation"],
                    json.dumps(vector, ensure_ascii=False),
                )

                cursor.execute(query, params)

            self.connection.commit()

        except Error as e:
            self.connection.rollback()
            print(f"Error inserting audio frame features: {e}")
            raise

        finally:
            cursor.close()

    def fetch_dataset_features(self) -> List[Dict[str, Any]]:
        query = """
            SELECT
                af.audio_id,
                af.file_name,
                af.file_path,
                feat.feature_id,
                feat.frame_index,
                feat.start_time,
                feat.end_time,
                feat.rms_energy,
                feat.zero_crossing_rate,
                feat.average_frequency,
                feat.frequency_variation,
                feat.average_pitch,
                feat.pitch_variation
            FROM audio_files af
            INNER JOIN audio_features feat ON af.audio_id = feat.audio_id
            WHERE af.dataset_type = 'dataset'
            ORDER BY af.audio_id, feat.frame_index
        """

        return self._execute(query, fetchall=True)

    def fetch_audio_by_id(self, audio_id: int) -> Optional[Dict[str, Any]]:
        query = """
            SELECT
                audio_id,
                file_name,
                file_path,
                dataset_type,
                duration_seconds,
                sample_rate,
                bit_depth,
                channels,
                file_size_bytes,
                file_format
            FROM audio_files
            WHERE audio_id = %s
        """

        return self._execute(query, (audio_id,), fetchone=True)