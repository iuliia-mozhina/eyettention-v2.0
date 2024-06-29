
import thread
from Queue import Queue as Queue
from threading import Semaphore
import time

class WorkerPool():
	active = 0
	queue = Queue()
	nproc = 0

	def __init__(self, nproc):
		self.nproc = nproc

		def consume_queue(worker_id):
			while True:
				job = self.queue.get(True)
				if job is None:
					self.queue.task_done()
					break
				job_func = job[0]
				job_kwargs = job[1]
				job_args = job[2] if type(job[2]) is tuple else (job[2], )
				job_wait_for_result_lock = job[3]
				self.active += 1
				job[4] = job_func(*job_args, **job_kwargs)
				if job_wait_for_result_lock is not None:
					job_wait_for_result_lock.release()
				self.active -= 1
				self.queue.task_done()
			self.nproc -= 1

		for i in xrange(self.nproc):
			thread.start_new_thread(consume_queue, (i,))

	def map(self, func, args):
		return self.do(func = func, args = [(x,) for x in args], n = len(args))
		
	def do(self, func, args = tuple(), kwargs = dict(), n = 1):
		lock = Semaphore(n)
		for i in xrange(n):
			lock.acquire(True)
		jobs = self.do_nowait(func = func, args = args, kwargs = kwargs, lock = lock, n = n)
		for i in xrange(n):
			lock.acquire(True)
		for i in xrange(n):
			lock.release()
		return [job[4] for job in jobs]

	def do_nowait(self, func, args = tuple(), kwargs = dict(), lock = None, n = 1):
		jobs = list()
		for i in xrange(n):
			job = [func[i] if type(func) is list else func, kwargs[i] if type(kwargs) is list else kwargs, args[i] if type(args) is list else args, lock, None]
			jobs.append(job)
			self.queue.put_nowait(job)
		return jobs

	def join(self):
		self.queue.join()

	def close(self):
		for i in xrange(self.nproc):
			self.queue.put_nowait(None)
		self.join()


