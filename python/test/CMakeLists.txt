set(PYTHON_TEST
	test_kernel_launch.py
	test_for.py
	test_reduce.py
	test_multithread.py
)

install(FILES ${PYTHON_TEST} DESTINATION test_python )