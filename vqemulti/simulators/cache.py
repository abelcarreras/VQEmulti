import pickle
import sqlite3
import numpy as np
from datetime import datetime
from vqemulti.utils import log_message
import zlib


class JobCache:
    """
    Class that makes use a SQLlite database to hadle data cache

    :param filename: name of the database file storing the cache data
    """

    def __init__(self, filename='jobs.db'):
        """
        Constructor
        """

        self._calculation_data_filename = filename

        conn = sqlite3.connect(self._calculation_data_filename)

        try:
            conn.execute('''CREATE TABLE DATA_TABLE
                                  (
                                      circuit_hash LONGTEXT,
                                      calc_type    TEXT,
                                      job_id       LONGTEXT,
                                      date         LONGTEXT
                                  );''')

            conn.commit()
            # print('Initialized database')

        except sqlite3.OperationalError as e:
            if str(e) != 'table DATA_TABLE already exists':
                raise e

        conn.close()


    def in_cache(self):
        if self._job_id is None:
            return False
        return True

    def get_job(self, circuit, mapped_observables=None):


        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()

        calc_type = 'sampler' if mapped_observables is not None else 'estimator'
        circuit_hash = self.get_hash(circuit, mapped_observables)

        job_id = self.retrieve_calculation_data(circuit_hash, calc_type)

        if job_id is None:
            return None

        job = service.job(job_id)

        if job.status() == 'CANCELLED':
            return None

        log_message('retrieved job ID: ', job_id, log_level=1)

        return service.job(job_id)


    def store_job(self, job, circuit, mapped_observables=None):

        try:
            job_id = job.job_id()
            print('status: ', job.status())
        except AttributeError:
            return

        log_message('job ID: ', job_id, log_level=1)

        date_time = datetime.now()

        conn = sqlite3.connect(self._calculation_data_filename)

        calc_type = 'sampler' if mapped_observables is not None else 'estimator'
        circuit_hash = self.get_hash(circuit, mapped_observables)

        conn.execute("INSERT into DATA_TABLE (circuit_hash, calc_type, job_id, date)  VALUES (?, ?, ?, ?)",
                           (circuit_hash,  calc_type, job_id, date_time))

        conn.commit()
        conn.close()


    @staticmethod
    def get_hash(circuit, mapped_observables=None):
        import hashlib

        if mapped_observables is None:
            pub_hash = hashlib.blake2b((str(circuit.draw(fold=-1))).encode(),
                                       digest_size=8,  # 64 bits
                                       ).hexdigest()
        else:
            pub_hash = hashlib.blake2b((repr(circuit.draw(fold=-1)) + repr(mapped_observables)).encode(),
                                       digest_size=8,  # 64 bits
                                       ).hexdigest()
        return pub_hash


    def retrieve_calculation_data(self, circuit_hash, calc_type):
        """
        retrieve calculation data from cache file

        :param input_qchem: QchemInput instance
        :param keyword: string that was used as a key to store the data
        :return:
        """
        conn = sqlite3.connect(self._calculation_data_filename)

        cursor = conn.execute("SELECT job_id FROM DATA_TABLE WHERE circuit_hash=? AND calc_type=?",
                                    (circuit_hash, calc_type))
        rows = cursor.fetchall()

        conn.close()

        return rows[0][0] if len(rows) > 0 else None

    def retrieve_calculation_data_from_id(self, id, keyword=None):
        """
        return data using database entry ID
        [Only for SQL database cache]

        :param id: databse entry ID
        :param keyword: string that was used as a key to store the data
        :return:
        """

        self._conn = sqlite3.connect(self._calculation_data_filename)

        if keyword is None:
            cursor = self._conn.execute("SELECT qcdata FROM DATA_TABLE WHERE input_hash=?", (id,))
            rows = cursor.fetchall()
        else:
            cursor = self._conn.execute("SELECT qcdata FROM DATA_TABLE WHERE input_hash=? AND parser=?",
                                        (id, keyword))
            rows = cursor.fetchall()

        self._conn.close()

        if len(rows) <= 0:
            return None
        elif len(rows) == 1:
            if self._compress:
                return pickle.loads(zlib.decompress(rows[0][0])) if len(rows) > 0 else None
            else:
                return pickle.loads(rows[0][0]) if len(rows) > 0 else None
        else:
            if self._compress:
                return [pickle.loads(zlib.decompress(r[0])) for r in rows]
            else:
                return [pickle.loads(r[0]) for r in rows]

    def list_database(self):
        """
        prints data inside database
        [Only for SQL databse cache]

        :return: None
        """
        conn = sqlite3.connect(self._calculation_data_filename)

        cursor = conn.execute("SELECT circuit_hash, calc_type, date from DATA_TABLE")


        print('{:^25} {:^25} {:^25}'.format('ID', 'CALC_TYPE', 'DATE'))
        print('--'*40)
        for row in cursor:
            try:
                print('{:<25} {:<25} {}'.format(*row))
            except IndexError:
                print('{:<25} {:<25}'.format(*row))

        conn.close()

    def integrity_check(self):
        """
        Check integrity of the database
        [Only for SQL databse cache]

        :return:
        """
        self._conn = sqlite3.connect(self._calculation_data_filename)

        cursor = self._conn.execute("PRAGMA integrity_check")
        rows = ''.join(*cursor.fetchall()[0])
        print(rows)
        self._conn.close()

        pass

    def fix_database(self, filename):
        """
        fix correupted database and store the recovered data in a new recovered database file
        [Only for SQL databse cache]

        :param filename: recovered database filename
        :return:
        """

        import subprocess, os

        dump_file = self._calculation_data_filename + '.dump'
        # dump_file = '_recovery.test'

        schema = subprocess.run(
            ['sqlite3',
             self._calculation_data_filename,
             '.output {}'.format(dump_file),
             '.dump',
             ],
            capture_output=True
        )

        with open(dump_file, 'r') as f:
            data = f.read().replace("ROLLBACK", "COMMIT")

        with open(dump_file, 'w') as f:
            f.write(data)

        try:
            os.remove(filename)
        except FileNotFoundError:
            pass

        schema = subprocess.run(
            ['sqlite3',
             filename,
             '.read {}'.format(dump_file)
             ],
            capture_output=True
        )

        os.remove(dump_file)

    def _recovery_kill(self, file):

        # cursor = self._conn.execute(".save ?", (file,))

        import subprocess
        schema = subprocess.run(
            ['sqlite3',
             self._calculation_data_filename,
             '.recover'.format(file)
             ],
            capture_output=True
        )

        print(schema.stdout)

    def get_all_data(self):
        """
        return a list of all data stored in the cache file

        :return: list of data
        """

        self._conn = sqlite3.connect(self._calculation_data_filename)

        cursor = self._conn.execute("SELECT * FROM DATA_TABLE")
        rows = cursor.fetchall()

        self._conn.close()

        calc_id_list = np.unique([r[0] for r in rows])

        calc_list = []
        for id in calc_id_list:
            data_dict = {}
            for r in rows:
                if r[0] == id:
                    if self._compress:
                        data_dict.update({r[1]: pickle.loads(zlib.decompress(r[2]))})
                    else:
                        data_dict.update({r[1]: pickle.loads(r[2])})
            calc_list.append(data_dict)

        return calc_list


if __name__ == '__main__':
    cache = JobCache()
    cache.list_database()
