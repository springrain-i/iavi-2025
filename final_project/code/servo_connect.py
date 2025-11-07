import serial
import serial.tools.list_ports
import time
import subprocess
import os
from pathlib import Path


class SerialPortManager:
    def __init__(self):
        self.occupied_ports = []

    def find_arduino_ports(self):
        """查找所有可能是Arduino的端口"""
        arduino_ports = []
        ports = serial.tools.list_ports.comports()

        for port in ports:
            # 检查常见Arduino描述
            if any(keyword in port.description.upper() for keyword in
                   ['ARDUINO', 'CH340', 'CP210', 'USB SERIAL', 'USB2.0']):
                arduino_ports.append(port.device)
                print(f"🔍 发现Arduino设备: {port.device} - {port.description}")

        return arduino_ports

    def kill_serial_processes(self, port=None):
        """终止可能占用串口的进程"""
        try:
            # Windows下查找占用COM端口的进程
            if os.name == 'nt':
                result = subprocess.run(
                    ['netstat', '-ano'],
                    capture_output=True,
                    text=True
                )

                lines = result.stdout.split('\n')
                for line in lines:
                    if 'COM' in line and (port is None or port in line):
                        parts = line.split()
                        if len(parts) >= 5:
                            pid = parts[-1]
                            # 终止进程
                            subprocess.run(['taskkill', '/F', '/PID', pid],
                                           capture_output=True)
                            print(f"✅ 已终止占用串口的进程 PID: {pid}")

            print("✅ 串口进程清理完成")
            return True

        except Exception as e:
            print(f"⚠️ 进程清理失败: {e}")
            return False

    def wait_for_port_ready(self, port, timeout=10):
        """等待端口就绪"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 尝试打开端口
                with serial.Serial(port, 9600, timeout=1) as ser:
                    ser.close()
                print(f"✅ 端口 {port} 就绪")
                return True
            except:
                print(f"⏳ 等待端口 {port} 就绪...")
                time.sleep(1)

        print(f"❌ 端口 {port} 超时未就绪")
        return False


class RobustArduinoFlasher:
    def __init__(self, port=None):
        self.port_manager = SerialPortManager()
        self.port = port or self._auto_detect_port()

    def _auto_detect_port(self):
        """自动检测Arduino端口"""
        print("🔍 自动检测Arduino端口...")
        arduino_ports = self.port_manager.find_arduino_ports()

        if not arduino_ports:
            print("❌ 未发现Arduino设备，请检查连接")
            return None

        if len(arduino_ports) == 1:
            selected_port = arduino_ports[0]
            print(f"✅ 自动选择端口: {selected_port}")
            return selected_port
        else:
            print("发现多个串口设备:")
            for i, port in enumerate(arduino_ports):
                print(f"  {i + 1}. {port}")

            try:
                choice = int(input("请选择端口编号: ")) - 1
                if 0 <= choice < len(arduino_ports):
                    return arduino_ports[choice]
            except:
                pass

            print("⚠️ 使用第一个端口")
            return arduino_ports[0]

    def prepare_for_flash(self):
        """准备烧录环境"""
        if not self.port:
            print("❌ 无可用端口")
            return False

        print(f"🔄 准备烧录环境，端口: {self.port}")

        # 1. 终止占用进程
        self.port_manager.kill_serial_processes(self.port)
        time.sleep(2)

        # 2. 等待端口就绪
        if not self.port_manager.wait_for_port_ready(self.port):
            return False

        # 3. 检查Arduino是否在bootloader模式
        if not self._check_arduino_ready():
            print("⚠️ 尝试重置Arduino...")
            self._reset_arduino()
            time.sleep(2)

        return True

    def _check_arduino_ready(self):
        """检查Arduino是否就绪"""
        try:
            with serial.Serial(self.port, 1200, timeout=1) as ser:
                time.sleep(0.5)
            return True
        except:
            return False

    def _reset_arduino(self):
        """尝试重置Arduino进入bootloader"""
        try:
            # 发送1200波特率信号触发重置
            with serial.Serial(self.port, 1200, timeout=1) as ser:
                time.sleep(0.5)
            print("✅ Arduino重置信号已发送")
            return True
        except Exception as e:
            print(f"❌ 重置失败: {e}")
            return False

    def generate_optimized_code(self, angle, sketch_name="ServoController"):
        """生成优化的Arduino代码，减少内存使用"""

        if angle > 0:
            code = f"""#include <Servo.h>  // 必须包含舵机库（原代码遗漏，会导致编译错误）
Servo myservo;  // 创建舵机对象
int pos = 0;    // 舵机位置变量

void setup() {{
  myservo.attach(9);  // 舵机信号线连接数字引脚9

  // 循环代码移到 setup() 内部（只执行一次）
  for (pos = 0; pos <= {angle}; pos += 1) {{
    myservo.write(pos);  // 设置舵机角度
    delay(7);            // 等待舵机到位（原注释写15ms，代码是7ms，可根据需要调整）
  }}
}}

void loop() {{
  // 这里留空，舵机转动一次后不再动作
}}
"""
        else:
            code = f"""#include <Servo.h>  // 必须包含舵机库（原代码遗漏，会导致编译错误）
Servo myservo;  // 创建舵机对象
int pos = 0;    // 舵机位置变量

void setup() {{
  myservo.attach(9);  // 舵机信号线连接数字引脚9

  // 循环代码移到 setup() 内部（只执行一次）
  for (pos = 0; pos >= {angle}; pos -= 1) {{
    myservo.write(pos);  // 设置舵机角度
    delay(7);            // 等待舵机到位（原注释写15ms，代码是7ms，可根据需要调整）
  }}
}}

void loop() {{
  // 这里留空，舵机转动一次后不再动作
}}
"""
        return code

    def flash_with_ide_direct(self, code, sketch_name="ServoController"):
        """直接使用Arduino IDE命令行烧录"""
        try:
            # 创建临时目录
            temp_dir = Path(f"C:/temp/arduino_upload/{sketch_name}")
            temp_dir.mkdir(parents=True, exist_ok=True)

            # 写入ino文件
            ino_file = temp_dir / f"{sketch_name}.ino"
            with open(ino_file, 'w', encoding='utf-8') as f:
                f.write(code)

            print(f"📁 代码文件: {ino_file}")

            # 查找Arduino IDE路径
            arduino_paths = [
                "D:\\arduino-ide_nightly-20241120_Windows_64bit\\Arduino IDE.exe",
                "C:\\Program Files (x86)\\Arduino\\arduino.exe",
                os.path.expanduser("~\\AppData\\Local\\Arduino\\arduino.exe")
            ]

            arduino_exe = None
            for path in arduino_paths:
                if os.path.exists(path):
                    arduino_exe = path
                    break

            if not arduino_exe:
                print("❌ 未找到Arduino IDE")
                return False

            # 构建烧录命令
            cmd = [
                arduino_exe,
                "--upload", str(ino_file),
                "--port", self.port,
                "--board", "arduino:avr:uno"
            ]

            print(f"🚀 执行烧录命令...")
            print(f"命令: {' '.join(cmd)}")

            # 执行烧录
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir  # 设置工作目录
            )

            if result.returncode == 0:
                print("✅ 烧录成功!")
                # 打印编译信息
                if "bytes" in result.stdout:
                    for line in result.stdout.split('\n'):
                        if "bytes" in line:
                            print(f"📊 {line.strip()}")
                return True
            else:
                print(f"❌ 烧录失败，返回码: {result.returncode}")
                print(f"错误输出: {result.stderr}")
                print(f"标准输出: {result.stdout}")
                return False

        except subprocess.TimeoutExpired:
            print("❌ 烧录超时")
            return False
        except Exception as e:
            print(f"❌ 烧录过程出错: {e}")
            return False

    def flash_with_alternative_methods(self, code, sketch_name="ServoController"):
        """尝试多种烧录方法"""
        methods = [
            self._try_arduino_cli,
            self._try_platformio,
            self._try_direct_avrdude
        ]

        for method in methods:
            print(f"\n🔄 尝试方法: {method.__name__}")
            if method(code, sketch_name):
                return True

        return False

    def _try_arduino_cli(self, code, sketch_name):
        """尝试使用arduino-cli"""
        try:
            # 创建临时目录
            temp_dir = Path("C:/temp/arduino_cli")
            temp_dir.mkdir(parents=True, exist_ok=True)

            ino_file = temp_dir / f"{sketch_name}.ino"
            with open(ino_file, 'w', encoding='utf-8') as f:
                f.write(code)

            # 初始化arduino-cli（如果需要）
            subprocess.run(["arduino-cli", "core", "update-index"],
                           capture_output=True, timeout=30)

            # 编译
            compile_cmd = [
                "arduino-cli", "compile",
                "--fqbn", "arduino:avr:uno",
                "--build-path", str(temp_dir / "build"),
                str(temp_dir)
            ]

            result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"编译失败: {result.stderr}")
                return False

            # 上传
            upload_cmd = [
                "arduino-cli", "upload",
                "--port", self.port,
                "--fqbn", "arduino:avr:uno",
                "--input-dir", str(temp_dir / "build"),
                str(temp_dir)
            ]

            result = subprocess.run(upload_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print("✅ arduino-cli烧录成功!")
                return True
            else:
                print(f"上传失败: {result.stderr}")
                return False

        except Exception as e:
            print(f"arduino-cli失败: {e}")
            return False

    def verify_upload(self, timeout=15):
        """验证上传是否成功"""
        print("🔍 验证上传结果...")

        try:
            # 等待Arduino重启
            time.sleep(3)

            with serial.Serial(self.port, 9600, timeout=1) as ser:
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if ser.in_waiting > 0:
                        line = ser.readline().decode().strip()
                        print(f"Arduino: {line}")
                        if "READY" in line:
                            print("✅ 验证成功! Arduino运行正常")
                            return True
                    time.sleep(0.1)

            print("⚠️ 未收到就绪信号，但可能已上传成功")
            return True

        except Exception as e:
            print(f"❌ 验证失败: {e}")
            return False


def robust_flash_workflow(angle):
    """健壮的烧录工作流程"""
    print("=== Arduino烧录解决方案 ===")

    # 初始化烧录器
    flasher = RobustArduinoFlasher()

    if not flasher.port:
        print("❌ 无法找到Arduino设备")
        return

    print(f"🎯 目标端口: {flasher.port}")

    # 步骤2: 准备烧录环境
    print("\n🔄 准备烧录环境...")
    if not flasher.prepare_for_flash():
        print("❌ 环境准备失败")
        return

    # 步骤3: 生成优化代码
    print("\n📄 生成优化代码...")
    code = flasher.generate_optimized_code(angle)
    print("✅ 代码生成完成")

    # 步骤4: 尝试烧录
    print("\n🚀 开始烧录...")

    # 首先尝试直接方法
    if flasher.flash_with_ide_direct(code):
        success = True
    else:
        print("\n🔄 尝试替代方法...")
        success = flasher.flash_with_alternative_methods(code)

    # 步骤5: 验证
    if success:
        print("\n✅ 烧录成功!")
        flasher.verify_upload()
    else:
        print("\n❌ 所有烧录方法都失败了")
        print("\n💡 建议:")
        print("1. 手动重启Arduino")
        print("2. 拔插USB线")
        print("3. 检查Arduino驱动")
        print("4. 尝试使用Arduino IDE手动上传")

robust_flash_workflow(8)