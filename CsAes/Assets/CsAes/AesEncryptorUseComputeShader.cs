using System;
using System.IO;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEditor;
using UnityEngine;

namespace CsAes
{
    /// <summary>
    /// コンピュートシェーダーで暗号処理を行うAES暗号クラス
    /// </summary>
    /// <remarks>
    /// 並列処理でかつ同じ処理で暗号/復号を行いたいのでCTRモードで使うことを前提としている
    /// </remarks>
    public class AesEncryptorUseComputeShader : IDisposable
    {
        // ラウンド回数
        private const int Nr = 10;

        // 鍵の長さ(バイト)
        private const int Nk = 4;

        // 1スレッドあたりが処理するバッファ長(uint型)
        // 8192 バイト処理するので int 型にするとして8912 / 4 = 2048 の配列要素を操作する
        private const int BufferSizeOneThread = 8192; // 8 KB / thread

        private const int BufferSizeOneThreadInt = BufferSizeOneThread / 4;

        // 合計スレッド数(並列処理数)
        private const int NumThread = 32; // 32並列

        // グループ数
        private const int NumGroup = 32;

        // 1回のディスパッチで処理できるバッファサイズ(byte型)
        // 並列処理数 × group数 × BUFFER_SIZE_ONE_THREAD_PER_UINT
        private const int BufferSizeOnceDispatch = BufferSizeOneThread * NumThread * NumGroup;

        // 一度のディスパッチで処理できるバッファサイズ(int型)
        // 8192 * 32 * 32 = 33554432 (約32MB)
        // 配列要素数 = 8388608
        private const int BufferSizeDispatchPerInt = BufferSizeOnceDispatch / 4;

        private readonly ComputeShader m_shader;
        private readonly ComputeBuffer m_workBuffer;
        private readonly ComputeBuffer m_nonceBuffer;
        private readonly ComputeBuffer m_counterBlockBuffer;
        private readonly ComputeBuffer m_expandKeys;
        private readonly int m_kernelIndex;
        private readonly int[] m_workArray = new int[BufferSizeDispatchPerInt];
        private readonly int[] m_roundKey = new int[Nk * (Nr + 1)];

        public AesEncryptorUseComputeShader()
        {
            m_shader = Resources.Load<ComputeShader>("Encrypt128");
            m_workBuffer = new ComputeBuffer(BufferSizeDispatchPerInt, sizeof(uint));
            m_kernelIndex = m_shader.FindKernel("Cipher");

            m_expandKeys = new(m_roundKey.Length, sizeof(uint));
            m_nonceBuffer = new ComputeBuffer(4, sizeof(uint));
            m_counterBlockBuffer = new ComputeBuffer(4, sizeof(uint));

            m_shader.SetBuffer(m_kernelIndex, "work", m_workBuffer);
            m_shader.SetBuffer(m_kernelIndex, "ekey", m_expandKeys);
            m_shader.SetBuffer(m_kernelIndex, "nonce", m_nonceBuffer);
            m_shader.SetBuffer(m_kernelIndex, "counterBlock", m_counterBlockBuffer);
            m_shader.SetInt("BufferSizeOneThreadInt", BufferSizeOneThreadInt);

            byte[] key = new byte[]
            {
                0x0,
                0x1,
                0x2,
                0x3,
                0x4,
                0x5,
                0x6,
                0x7,
                0x8,
                0x9,
                0xA,
                0xB,
                0xC,
                0xD,
                0xE,
                0xF
            };
            ResetRoundKeys(key);
        }

        private void ResetRoundKeys(byte[] key)
        {
            int num1 = 0;
            for (int index1 = 0; index1 < Nk; ++index1)
            {
                byte[] numArray2 = key;
                int index2 = num1;
                int num2 = index2 + 1;
                int num3 = (int) ((uint) numArray2[index2] << 24);
                int index3 = num2;
                int num4 = index3 + 1;
                int num5 = key[index3] << 16;
                int num6 = num3 | num5;
                int index4 = num4;
                int num7 = index4 + 1;
                int num8 = key[index4] << 8;
                int num9 = num6 | num8;
                int index5 = num7;
                num1 = index5 + 1;
                int num10 = key[index5];
                int num11 = (num9 | num10);
                m_roundKey[index1] = num11;
            }

            for (int nk = Nk; nk < m_roundKey.Length; ++nk)
            {
                int a = m_roundKey[nk - 1];
                if (nk % Nk == 0)
                {
                    int rotByte = AesUtility.RotByte(a);
                    int rcon = (int) AesUtility.RCon[nk / Nk];
                    a = AesUtility.SubByte(rotByte ^ rcon);
                }

                m_roundKey[nk] = m_roundKey[nk - Nk] ^ a;
            }

            m_expandKeys.SetData(m_roundKey);
        }

        private unsafe void CopyToWork(byte* pBuffer, int length)
        {
            fixed (int* pWork = &m_workArray[0])
            {
                byte* pWorkByte = (byte*) pWork;
                for (int i = 0; i < length; ++i)
                {
                    pWorkByte[i] = pBuffer[0];
                }
            }
        }

        private unsafe void CopyToBuffer(byte* pBuffer, int length)
        {
            fixed (int* pWork = &m_workArray[0])
            {
                byte* pWorkByte = (byte*) pWork;
                for (int i = 0; i < length; ++i)
                {
                    pBuffer[0] = pWorkByte[i];
                }
            }
        }

        public unsafe void Cipher(NativeArray<byte> ioBuffer, int startIndex, int length)
        {
            byte* pointer = (byte*) ioBuffer.GetUnsafePtr();
            pointer += startIndex;
            CopyToWork(pointer, length);
            m_workBuffer.SetData(m_workArray);
            m_shader.SetInt("blockIndex", (startIndex / 16) + 1);
            m_shader.Dispatch(m_kernelIndex, NumGroup, 1, 1);
            m_workBuffer.GetData(m_workArray);
            CopyToBuffer(pointer, length);
        }

        public void Dispose()
        {
            m_workBuffer.Dispose();
            m_expandKeys.Dispose();
            m_nonceBuffer.Dispose();
            m_counterBlockBuffer.Dispose();
            Resources.UnloadAsset(m_shader);
        }

        [MenuItem("Aes/ComputeShader/Test Encrypt")]
        public static void TestEncrypt()
        {
            byte[] bytes = File.ReadAllBytes("ExampleFile/Example1.png");
            NativeArray<byte> array = new(bytes, Allocator.Persistent);
            using (AesEncryptorUseComputeShader instance = new())
            {
                Debug.Log("Encryption start");

                DateTime start = DateTime.Now;
                int position = 0;
                while ((array.Length - position) != 0)
                {
                    int readCount = array.Length - position;
                    if (readCount >= BufferSizeOnceDispatch)
                    {
                        readCount = BufferSizeOnceDispatch;
                    }

                    instance.Cipher(array, position, readCount);
                    position += readCount;
                }

                DateTime finished = DateTime.Now;
                Debug.Log($"Encryption finished time:{(finished - start).TotalSeconds}");
                File.WriteAllBytes("ExampleFile/encrypted.enc", array.ToArray());
                array.Dispose();
            }
        }

        [MenuItem("Aes/ComputeShader/Test Decrypt")]
        public static void TestDecrypt()
        {
            byte[] bytes = File.ReadAllBytes("ExampleFile/encrypted.enc");
            NativeArray<byte> array = new(bytes, Allocator.Persistent);
            using (AesEncryptorUseComputeShader instance = new())
            {
                Debug.Log("Encryption start");

                DateTime start = DateTime.Now;
                int position = 0;
                while ((array.Length - position) != 0)
                {
                    int readCount = array.Length - position;
                    if (readCount >= BufferSizeOnceDispatch)
                    {
                        readCount = BufferSizeOnceDispatch;
                    }

                    instance.Cipher(array, position, readCount);
                    position += readCount;
                }

                DateTime finished = DateTime.Now;
                Debug.Log($"Encryption finished time:{(finished - start).TotalSeconds}");
                File.WriteAllBytes("ExampleFile/decrypted.png", array.ToArray());
                array.Dispose();
            }
        }
    }
}
