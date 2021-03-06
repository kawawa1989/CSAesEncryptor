using System;
using UnityEngine;

namespace CsAes
{
    public class AesUtility : IDisposable
    {
        public static readonly uint[] RCon = new uint[]
        {
            0U,
            16777216U,
            33554432U,
            67108864U,
            134217728U,
            268435456U,
            536870912U,
            1073741824U,
            2147483648U,
            452984832U,
            905969664U,
            1811939328U,
            3623878656U,
            2868903936U,
            1291845632U,
            2583691264U,
            788529152U,
            1577058304U,
            3154116608U,
            1660944384U,
            3321888768U,
            2533359616U,
            889192448U,
            1778384896U,
            3556769792U,
            3003121664U,
            2097152000U,
            4194304000U,
            4009754624U,
            3305111552U
        };

        private static readonly uint[] SBox =
        {
            99,
            124,
            119,
            123,
            242,
            107,
            111,
            197,
            48,
            1,
            103,
            43,
            254,
            215,
            171,
            118,
            202,
            130,
            201,
            125,
            250,
            89,
            71,
            240,
            173,
            212,
            162,
            175,
            156,
            164,
            114,
            192,
            183,
            253,
            147,
            38,
            54,
            63,
            247,
            204,
            52,
            165,
            229,
            241,
            113,
            216,
            49,
            21,
            4,
            199,
            35,
            195,
            24,
            150,
            5,
            154,
            7,
            18,
            128,
            226,
            235,
            39,
            178,
            117,
            9,
            131,
            44,
            26,
            27,
            110,
            90,
            160,
            82,
            59,
            214,
            179,
            41,
            227,
            47,
            132,
            83,
            209,
            0,
            237,
            32,
            252,
            177,
            91,
            106,
            203,
            190,
            57,
            74,
            76,
            88,
            207,
            208,
            239,
            170,
            251,
            67,
            77,
            51,
            133,
            69,
            249,
            2,
            127,
            80,
            60,
            159,
            168,
            81,
            163,
            64,
            143,
            146,
            157,
            56,
            245,
            188,
            182,
            218,
            33,
            16,
            byte.MaxValue,
            243,
            210,
            205,
            12,
            19,
            236,
            95,
            151,
            68,
            23,
            196,
            167,
            126,
            61,
            100,
            93,
            25,
            115,
            96,
            129,
            79,
            220,
            34,
            42,
            144,
            136,
            70,
            238,
            184,
            20,
            222,
            94,
            11,
            219,
            224,
            50,
            58,
            10,
            73,
            6,
            36,
            92,
            194,
            211,
            172,
            98,
            145,
            149,
            228,
            121,
            231,
            200,
            55,
            109,
            141,
            213,
            78,
            169,
            108,
            86,
            244,
            234,
            101,
            122,
            174,
            8,
            186,
            120,
            37,
            46,
            28,
            166,
            180,
            198,
            232,
            221,
            116,
            31,
            75,
            189,
            139,
            138,
            112,
            62,
            181,
            102,
            72,
            3,
            246,
            14,
            97,
            53,
            87,
            185,
            134,
            193,
            29,
            158,
            225,
            248,
            152,
            17,
            105,
            217,
            142,
            148,
            155,
            30,
            135,
            233,
            206,
            85,
            40,
            223,
            140,
            161,
            137,
            13,
            191,
            230,
            66,
            104,
            65,
            153,
            45,
            15,
            176,
            84,
            187,
            22
        };

        private static readonly uint[] T0 = new uint[]
        {
            3328402341U,
            4168907908U,
            4000806809U,
            4135287693U,
            4294111757U,
            3597364157U,
            3731845041U,
            2445657428U,
            1613770832U,
            33620227U,
            3462883241U,
            1445669757U,
            3892248089U,
            3050821474U,
            1303096294U,
            3967186586U,
            2412431941U,
            528646813U,
            2311702848U,
            4202528135U,
            4026202645U,
            2992200171U,
            2387036105U,
            4226871307U,
            1101901292U,
            3017069671U,
            1604494077U,
            1169141738U,
            597466303U,
            1403299063U,
            3832705686U,
            2613100635U,
            1974974402U,
            3791519004U,
            1033081774U,
            1277568618U,
            1815492186U,
            2118074177U,
            4126668546U,
            2211236943U,
            1748251740U,
            1369810420U,
            3521504564U,
            4193382664U,
            3799085459U,
            2883115123U,
            1647391059U,
            706024767U,
            134480908U,
            2512897874U,
            1176707941U,
            2646852446U,
            806885416U,
            932615841U,
            168101135U,
            798661301U,
            235341577U,
            605164086U,
            461406363U,
            3756188221U,
            3454790438U,
            1311188841U,
            2142417613U,
            3933566367U,
            302582043U,
            495158174U,
            1479289972U,
            874125870U,
            907746093U,
            3698224818U,
            3025820398U,
            1537253627U,
            2756858614U,
            1983593293U,
            3084310113U,
            2108928974U,
            1378429307U,
            3722699582U,
            1580150641U,
            327451799U,
            2790478837U,
            3117535592U,
            0U,
            3253595436U,
            1075847264U,
            3825007647U,
            2041688520U,
            3059440621U,
            3563743934U,
            2378943302U,
            1740553945U,
            1916352843U,
            2487896798U,
            2555137236U,
            2958579944U,
            2244988746U,
            3151024235U,
            3320835882U,
            1336584933U,
            3992714006U,
            2252555205U,
            2588757463U,
            1714631509U,
            293963156U,
            2319795663U,
            3925473552U,
            67240454U,
            4269768577U,
            2689618160U,
            2017213508U,
            631218106U,
            1269344483U,
            2723238387U,
            1571005438U,
            2151694528U,
            93294474U,
            1066570413U,
            563977660U,
            1882732616U,
            4059428100U,
            1673313503U,
            2008463041U,
            2950355573U,
            1109467491U,
            537923632U,
            3858759450U,
            4260623118U,
            3218264685U,
            2177748300U,
            403442708U,
            638784309U,
            3287084079U,
            3193921505U,
            899127202U,
            2286175436U,
            773265209U,
            2479146071U,
            1437050866U,
            4236148354U,
            2050833735U,
            3362022572U,
            3126681063U,
            840505643U,
            3866325909U,
            3227541664U,
            427917720U,
            2655997905U,
            2749160575U,
            1143087718U,
            1412049534U,
            999329963U,
            193497219U,
            2353415882U,
            3354324521U,
            1807268051U,
            672404540U,
            2816401017U,
            3160301282U,
            369822493U,
            2916866934U,
            3688947771U,
            1681011286U,
            1949973070U,
            336202270U,
            2454276571U,
            201721354U,
            1210328172U,
            3093060836U,
            2680341085U,
            3184776046U,
            1135389935U,
            3294782118U,
            965841320U,
            831886756U,
            3554993207U,
            4068047243U,
            3588745010U,
            2345191491U,
            1849112409U,
            3664604599U,
            26054028U,
            2983581028U,
            2622377682U,
            1235855840U,
            3630984372U,
            2891339514U,
            4092916743U,
            3488279077U,
            3395642799U,
            4101667470U,
            1202630377U,
            268961816U,
            1874508501U,
            4034427016U,
            1243948399U,
            1546530418U,
            941366308U,
            1470539505U,
            1941222599U,
            2546386513U,
            3421038627U,
            2715671932U,
            3899946140U,
            1042226977U,
            2521517021U,
            1639824860U,
            227249030U,
            260737669U,
            3765465232U,
            2084453954U,
            1907733956U,
            3429263018U,
            2420656344U,
            100860677U,
            4160157185U,
            470683154U,
            3261161891U,
            1781871967U,
            2924959737U,
            1773779408U,
            394692241U,
            2579611992U,
            974986535U,
            664706745U,
            3655459128U,
            3958962195U,
            731420851U,
            571543859U,
            3530123707U,
            2849626480U,
            126783113U,
            865375399U,
            765172662U,
            1008606754U,
            361203602U,
            3387549984U,
            2278477385U,
            2857719295U,
            1344809080U,
            2782912378U,
            59542671U,
            1503764984U,
            160008576U,
            437062935U,
            1707065306U,
            3622233649U,
            2218934982U,
            3496503480U,
            2185314755U,
            697932208U,
            1512910199U,
            504303377U,
            2075177163U,
            2824099068U,
            1841019862U,
            739644986U
        };

        private static readonly uint[] T1 = new uint[]
        {
            2781242211U,
            2230877308U,
            2582542199U,
            2381740923U,
            234877682U,
            3184946027U,
            2984144751U,
            1418839493U,
            1348481072U,
            50462977U,
            2848876391U,
            2102799147U,
            434634494U,
            1656084439U,
            3863849899U,
            2599188086U,
            1167051466U,
            2636087938U,
            1082771913U,
            2281340285U,
            368048890U,
            3954334041U,
            3381544775U,
            201060592U,
            3963727277U,
            1739838676U,
            4250903202U,
            3930435503U,
            3206782108U,
            4149453988U,
            2531553906U,
            1536934080U,
            3262494647U,
            484572669U,
            2923271059U,
            1783375398U,
            1517041206U,
            1098792767U,
            49674231U,
            1334037708U,
            1550332980U,
            4098991525U,
            886171109U,
            150598129U,
            2481090929U,
            1940642008U,
            1398944049U,
            1059722517U,
            201851908U,
            1385547719U,
            1699095331U,
            1587397571U,
            674240536U,
            2704774806U,
            252314885U,
            3039795866U,
            151914247U,
            908333586U,
            2602270848U,
            1038082786U,
            651029483U,
            1766729511U,
            3447698098U,
            2682942837U,
            454166793U,
            2652734339U,
            1951935532U,
            775166490U,
            758520603U,
            3000790638U,
            4004797018U,
            4217086112U,
            4137964114U,
            1299594043U,
            1639438038U,
            3464344499U,
            2068982057U,
            1054729187U,
            1901997871U,
            2534638724U,
            4121318227U,
            1757008337U,
            0U,
            750906861U,
            1614815264U,
            535035132U,
            3363418545U,
            3988151131U,
            3201591914U,
            1183697867U,
            3647454910U,
            1265776953U,
            3734260298U,
            3566750796U,
            3903871064U,
            1250283471U,
            1807470800U,
            717615087U,
            3847203498U,
            384695291U,
            3313910595U,
            3617213773U,
            1432761139U,
            2484176261U,
            3481945413U,
            283769337U,
            100925954U,
            2180939647U,
            4037038160U,
            1148730428U,
            3123027871U,
            3813386408U,
            4087501137U,
            4267549603U,
            3229630528U,
            2315620239U,
            2906624658U,
            3156319645U,
            1215313976U,
            82966005U,
            3747855548U,
            3245848246U,
            1974459098U,
            1665278241U,
            807407632U,
            451280895U,
            251524083U,
            1841287890U,
            1283575245U,
            337120268U,
            891687699U,
            801369324U,
            3787349855U,
            2721421207U,
            3431482436U,
            959321879U,
            1469301956U,
            4065699751U,
            2197585534U,
            1199193405U,
            2898814052U,
            3887750493U,
            724703513U,
            2514908019U,
            2696962144U,
            2551808385U,
            3516813135U,
            2141445340U,
            1715741218U,
            2119445034U,
            2872807568U,
            2198571144U,
            3398190662U,
            700968686U,
            3547052216U,
            1009259540U,
            2041044702U,
            3803995742U,
            487983883U,
            1991105499U,
            1004265696U,
            1449407026U,
            1316239930U,
            504629770U,
            3683797321U,
            168560134U,
            1816667172U,
            3837287516U,
            1570751170U,
            1857934291U,
            4014189740U,
            2797888098U,
            2822345105U,
            2754712981U,
            936633572U,
            2347923833U,
            852879335U,
            1133234376U,
            1500395319U,
            3084545389U,
            2348912013U,
            1689376213U,
            3533459022U,
            3762923945U,
            3034082412U,
            4205598294U,
            133428468U,
            634383082U,
            2949277029U,
            2398386810U,
            3913789102U,
            403703816U,
            3580869306U,
            2297460856U,
            1867130149U,
            1918643758U,
            607656988U,
            4049053350U,
            3346248884U,
            1368901318U,
            600565992U,
            2090982877U,
            2632479860U,
            557719327U,
            3717614411U,
            3697393085U,
            2249034635U,
            2232388234U,
            2430627952U,
            1115438654U,
            3295786421U,
            2865522278U,
            3633334344U,
            84280067U,
            33027830U,
            303828494U,
            2747425121U,
            1600795957U,
            4188952407U,
            3496589753U,
            2434238086U,
            1486471617U,
            658119965U,
            3106381470U,
            953803233U,
            334231800U,
            3005978776U,
            857870609U,
            3151128937U,
            1890179545U,
            2298973838U,
            2805175444U,
            3056442267U,
            574365214U,
            2450884487U,
            550103529U,
            1233637070U,
            4289353045U,
            2018519080U,
            2057691103U,
            2399374476U,
            4166623649U,
            2148108681U,
            387583245U,
            3664101311U,
            836232934U,
            3330556482U,
            3100665960U,
            3280093505U,
            2955516313U,
            2002398509U,
            287182607U,
            3413881008U,
            4238890068U,
            3597515707U,
            975967766U
        };

        private static readonly uint[] T2 = new uint[]
        {
            1671808611U,
            2089089148U,
            2006576759U,
            2072901243U,
            4061003762U,
            1807603307U,
            1873927791U,
            3310653893U,
            810573872U,
            16974337U,
            1739181671U,
            729634347U,
            4263110654U,
            3613570519U,
            2883997099U,
            1989864566U,
            3393556426U,
            2191335298U,
            3376449993U,
            2106063485U,
            4195741690U,
            1508618841U,
            1204391495U,
            4027317232U,
            2917941677U,
            3563566036U,
            2734514082U,
            2951366063U,
            2629772188U,
            2767672228U,
            1922491506U,
            3227229120U,
            3082974647U,
            4246528509U,
            2477669779U,
            644500518U,
            911895606U,
            1061256767U,
            4144166391U,
            3427763148U,
            878471220U,
            2784252325U,
            3845444069U,
            4043897329U,
            1905517169U,
            3631459288U,
            827548209U,
            356461077U,
            67897348U,
            3344078279U,
            593839651U,
            3277757891U,
            405286936U,
            2527147926U,
            84871685U,
            2595565466U,
            118033927U,
            305538066U,
            2157648768U,
            3795705826U,
            3945188843U,
            661212711U,
            2999812018U,
            1973414517U,
            152769033U,
            2208177539U,
            745822252U,
            439235610U,
            455947803U,
            1857215598U,
            1525593178U,
            2700827552U,
            1391895634U,
            994932283U,
            3596728278U,
            3016654259U,
            695947817U,
            3812548067U,
            795958831U,
            2224493444U,
            1408607827U,
            3513301457U,
            0U,
            3979133421U,
            543178784U,
            4229948412U,
            2982705585U,
            1542305371U,
            1790891114U,
            3410398667U,
            3201918910U,
            961245753U,
            1256100938U,
            1289001036U,
            1491644504U,
            3477767631U,
            3496721360U,
            4012557807U,
            2867154858U,
            4212583931U,
            1137018435U,
            1305975373U,
            861234739U,
            2241073541U,
            1171229253U,
            4178635257U,
            33948674U,
            2139225727U,
            1357946960U,
            1011120188U,
            2679776671U,
            2833468328U,
            1374921297U,
            2751356323U,
            1086357568U,
            2408187279U,
            2460827538U,
            2646352285U,
            944271416U,
            4110742005U,
            3168756668U,
            3066132406U,
            3665145818U,
            560153121U,
            271589392U,
            4279952895U,
            4077846003U,
            3530407890U,
            3444343245U,
            202643468U,
            322250259U,
            3962553324U,
            1608629855U,
            2543990167U,
            1154254916U,
            389623319U,
            3294073796U,
            2817676711U,
            2122513534U,
            1028094525U,
            1689045092U,
            1575467613U,
            422261273U,
            1939203699U,
            1621147744U,
            2174228865U,
            1339137615U,
            3699352540U,
            577127458U,
            712922154U,
            2427141008U,
            2290289544U,
            1187679302U,
            3995715566U,
            3100863416U,
            339486740U,
            3732514782U,
            1591917662U,
            186455563U,
            3681988059U,
            3762019296U,
            844522546U,
            978220090U,
            169743370U,
            1239126601U,
            101321734U,
            611076132U,
            1558493276U,
            3260915650U,
            3547250131U,
            2901361580U,
            1655096418U,
            2443721105U,
            2510565781U,
            3828863972U,
            2039214713U,
            3878868455U,
            3359869896U,
            928607799U,
            1840765549U,
            2374762893U,
            3580146133U,
            1322425422U,
            2850048425U,
            1823791212U,
            1459268694U,
            4094161908U,
            3928346602U,
            1706019429U,
            2056189050U,
            2934523822U,
            135794696U,
            3134549946U,
            2022240376U,
            628050469U,
            779246638U,
            472135708U,
            2800834470U,
            3032970164U,
            3327236038U,
            3894660072U,
            3715932637U,
            1956440180U,
            522272287U,
            1272813131U,
            3185336765U,
            2340818315U,
            2323976074U,
            1888542832U,
            1044544574U,
            3049550261U,
            1722469478U,
            1222152264U,
            50660867U,
            4127324150U,
            236067854U,
            1638122081U,
            895445557U,
            1475980887U,
            3117443513U,
            2257655686U,
            3243809217U,
            489110045U,
            2662934430U,
            3778599393U,
            4162055160U,
            2561878936U,
            288563729U,
            1773916777U,
            3648039385U,
            2391345038U,
            2493985684U,
            2612407707U,
            505560094U,
            2274497927U,
            3911240169U,
            3460925390U,
            1442818645U,
            678973480U,
            3749357023U,
            2358182796U,
            2717407649U,
            2306869641U,
            219617805U,
            3218761151U,
            3862026214U,
            1120306242U,
            1756942440U,
            1103331905U,
            2578459033U,
            762796589U,
            252780047U,
            2966125488U,
            1425844308U,
            3151392187U,
            372911126U
        };

        private static readonly uint[] T3 = new uint[]
        {
            1667474886U,
            2088535288U,
            2004326894U,
            2071694838U,
            4075949567U,
            1802223062U,
            1869591006U,
            3318043793U,
            808472672U,
            16843522U,
            1734846926U,
            724270422U,
            4278065639U,
            3621216949U,
            2880169549U,
            1987484396U,
            3402253711U,
            2189597983U,
            3385409673U,
            2105378810U,
            4210693615U,
            1499065266U,
            1195886990U,
            4042263547U,
            2913856577U,
            3570689971U,
            2728590687U,
            2947541573U,
            2627518243U,
            2762274643U,
            1920112356U,
            3233831835U,
            3082273397U,
            4261223649U,
            2475929149U,
            640051788U,
            909531756U,
            1061110142U,
            4160160501U,
            3435941763U,
            875846760U,
            2779116625U,
            3857003729U,
            4059105529U,
            1903268834U,
            3638064043U,
            825316194U,
            353713962U,
            67374088U,
            3351728789U,
            589522246U,
            3284360861U,
            404236336U,
            2526454071U,
            84217610U,
            2593830191U,
            117901582U,
            303183396U,
            2155911963U,
            3806477791U,
            3958056653U,
            656894286U,
            2998062463U,
            1970642922U,
            151591698U,
            2206440989U,
            741110872U,
            437923380U,
            454765878U,
            1852748508U,
            1515908788U,
            2694904667U,
            1381168804U,
            993742198U,
            3604373943U,
            3014905469U,
            690584402U,
            3823320797U,
            791638366U,
            2223281939U,
            1398011302U,
            3520161977U,
            0U,
            3991743681U,
            538992704U,
            4244381667U,
            2981218425U,
            1532751286U,
            1785380564U,
            3419096717U,
            3200178535U,
            960056178U,
            1246420628U,
            1280103576U,
            1482221744U,
            3486468741U,
            3503319995U,
            4025428677U,
            2863326543U,
            4227536621U,
            1128514950U,
            1296947098U,
            859002214U,
            2240123921U,
            1162203018U,
            4193849577U,
            33687044U,
            2139062782U,
            1347481760U,
            1010582648U,
            2678045221U,
            2829640523U,
            1364325282U,
            2745433693U,
            1077985408U,
            2408548869U,
            2459086143U,
            2644360225U,
            943212656U,
            4126475505U,
            3166494563U,
            3065430391U,
            3671750063U,
            555836226U,
            269496352U,
            4294908645U,
            4092792573U,
            3537006015U,
            3452783745U,
            202118168U,
            320025894U,
            3974901699U,
            1600119230U,
            2543297077U,
            1145359496U,
            387397934U,
            3301201811U,
            2812801621U,
            2122220284U,
            1027426170U,
            1684319432U,
            1566435258U,
            421079858U,
            1936954854U,
            1616945344U,
            2172753945U,
            1330631070U,
            3705438115U,
            572679748U,
            707427924U,
            2425400123U,
            2290647819U,
            1179044492U,
            4008585671U,
            3099120491U,
            336870440U,
            3739122087U,
            1583276732U,
            185277718U,
            3688593069U,
            3772791771U,
            842159716U,
            976899700U,
            168435220U,
            1229577106U,
            101059084U,
            606366792U,
            1549591736U,
            3267517855U,
            3553849021U,
            2897014595U,
            1650632388U,
            2442242105U,
            2509612081U,
            3840161747U,
            2038008818U,
            3890688725U,
            3368567691U,
            926374254U,
            1835907034U,
            2374863873U,
            3587531953U,
            1313788572U,
            2846482505U,
            1819063512U,
            1448540844U,
            4109633523U,
            3941213647U,
            1701162954U,
            2054852340U,
            2930698567U,
            134748176U,
            3132806511U,
            2021165296U,
            623210314U,
            774795868U,
            471606328U,
            2795958615U,
            3031746419U,
            3334885783U,
            3907527627U,
            3722280097U,
            1953799400U,
            522133822U,
            1263263126U,
            3183336545U,
            2341176845U,
            2324333839U,
            1886425312U,
            1044267644U,
            3048588401U,
            1718004428U,
            1212733584U,
            50529542U,
            4143317495U,
            235803164U,
            1633788866U,
            892690282U,
            1465383342U,
            3115962473U,
            2256965911U,
            3250673817U,
            488449850U,
            2661202215U,
            3789633753U,
            4177007595U,
            2560144171U,
            286339874U,
            1768537042U,
            3654906025U,
            2391705863U,
            2492770099U,
            2610673197U,
            505291324U,
            2273808917U,
            3924369609U,
            3469625735U,
            1431699370U,
            673740880U,
            3755965093U,
            2358021891U,
            2711746649U,
            2307489801U,
            218961690U,
            3217021541U,
            3873845719U,
            1111672452U,
            1751693520U,
            1094828930U,
            2576986153U,
            757954394U,
            252645662U,
            2964376443U,
            1414855848U,
            3149649517U,
            370555436U
        };

        public static int SubByte(int a)
        {
            int index1 = byte.MaxValue & a;
            int num1 = (int) SBox[index1];
            int index2 = byte.MaxValue & a >> 8;
            int num2 = (int) SBox[index2] << 8;
            int num3 = num1 | num2;
            int index3 = byte.MaxValue & a >> 16;
            int num4 = (int) SBox[index3] << 16;
            int num5 = num3 | num4;
            int index4 = byte.MaxValue & a >> 24;
            int num6 = (int) SBox[index4] << 24;
            return num5 | num6;
        }

        public static int RotByte(int a)
        {
            return ((a << 8) | (a >> 24)) & byte.MaxValue;
        }

        private readonly ComputeBuffer m_t0 = new(256, sizeof(uint));
        private readonly ComputeBuffer m_t1 = new(256, sizeof(uint));
        private readonly ComputeBuffer m_t2 = new(256, sizeof(uint));
        private readonly ComputeBuffer m_t3 = new(256, sizeof(uint));
        private readonly ComputeBuffer m_sBox = new(256, sizeof(uint));
        private bool m_disposed;

        public AesUtility()
        {
            m_t0.SetData(T0);
            Shader.SetGlobalBuffer(Shader.PropertyToID("T0"), m_t0);

            m_t1.SetData(T1);
            Shader.SetGlobalBuffer(Shader.PropertyToID("T1"), m_t1);

            m_t2.SetData(T2);
            Shader.SetGlobalBuffer(Shader.PropertyToID("T2"), m_t2);

            m_t3.SetData(T3);
            Shader.SetGlobalBuffer(Shader.PropertyToID("T3"), m_t3);

            m_sBox.SetData(SBox);
            Shader.SetGlobalBuffer(Shader.PropertyToID("sBox"), m_sBox);
        }

        public void Dispose()
        {
            if (m_disposed)
            {
                return;
            }

            m_disposed = true;
            m_t0.Dispose();
            m_t1.Dispose();
            m_t2.Dispose();
            m_t3.Dispose();
            m_sBox.Dispose();
        }
    }
}