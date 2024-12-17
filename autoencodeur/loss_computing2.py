import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 3),
            nn.ReLU(),
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 100),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


List_of_CVs = [('4e1ad3a9-c947-4a99-9bbe-6a01c669bc53', 1.4), ('f1aa0c14-7014-4491-acff-82953c519abc', 1.38), ('5e3b87d4-640f-4234-9b20-60620429cbf3', 1.08), ('8ebb2dbe-f83d-42d1-86e2-073392f7a343', 1.06), ('455875d0-b14e-47ed-a8f2-f3bdd9ae23ff', 1.04), ('e66f2425-2499-493f-a5a4-ff3b2d3eff51', 1.02), ('8e18b765-f76a-44ff-956e-5af377d94621', 1.02), ('49594d0d-8d1f-4add-a559-f9b63f740037', 0.95), ('0510b4df-b0be-4dc4-b81e-2b288516fd6b', 0.93), ('67ddd3a1-d630-49b1-83aa-2f87053bd513', 0.89), ('9e3fd9ab-dbe6-4b83-b367-11df51d6512d', 0.85), ('61fbaddf-5461-4f2e-bb49-1abe9c77aba3', 0.85), ('3f8809cd-3de4-4a98-bc4f-752213b8af39', 0.85), ('7f48cf4d-daa3-499b-9866-9caa128805bf', 0.82), ('61d1dfad-5c94-45c5-9618-4f92f69ccc4e', 0.8), ('ab39b18a-9ad9-413b-b436-9678d4be4bc0', 0.79), ('6dbc170f-b1a5-4668-8004-cc7aa9fbdc2a', 0.79), ('5839e957-47b5-4628-8430-03a1ef611a2c', 0.78), ('5416bd19-2830-4c81-bca9-63c3ab7fefce', 0.77), ('d63f2c8b-8be7-4809-b496-b0fa463549bb', 0.75), ('c3f68b78-e1b5-475e-a606-dd00fcb95fa7', 0.74), ('5cfd2ab4-074c-4cf6-91e2-0beb5b5cfcbe', 0.71), ('2ad4c9f5-34b7-4ab4-8e13-6ef548cad5e5', 0.71), ('70c38b57-79fb-4323-b92d-fdeccbdf05cc', 0.65), ('aef7a909-6f1f-4ae5-9d03-f3e17b28344f', 0.64), ('654a3bb7-5669-4d42-b37d-50e153a9d1e4', 0.64), ('166ceebe-f42f-49a1-bf42-73697942238e', 0.6), ('50de6e02-c915-4fc6-aa26-fe2ba6598c57', 0.58), ('de7e208c-cd6c-4946-b384-74064878123c', 0.56), ('69f225de-5d9b-4bff-9214-dbb0881a31ce', 0.56), ('32be0c37-a9b1-4de2-a17a-bba8a87b61d0', 0.54), ('6304483c-ddd0-428f-b43b-20c1a0cef904', 0.53), ('1200c9e3-091c-4634-af71-0503567b2271', 0.51), ('4ee5f1a2-042f-43c0-851d-f45f5fb05dac', 0.49), ('09c7950a-0b23-4d1c-91f3-502c8a14444a', 0.47), ('aaf98747-fa11-4aa0-a4f0-40b23c6b09c3', 0.4), ('c9490ca9-16b9-43df-aa26-57929690204e', 0.31), ('7ed332ac-a1bb-4c7c-8ec6-98c6b61c2839', 0.31), ('d42e0389-aac7-4f49-b7a5-76e98a4f1965', 0.29), ('60628ed7-880b-4db1-a847-b9480235b20b', 0.29), ('df1e5013-9945-4691-99c7-e714a4d2d16c', 0.28), ('d2d0a2f9-cff1-4aad-ae7c-499d97670638', 0.28), ('9aa9d2be-73bc-4baa-848a-44c78d6b4845', 0.28), ('1f5a4f5d-b1cc-483c-af73-97df01729784', 0.28), ('d74e72a4-40bb-493e-a8ac-eeebb9033506', 0.27), ('8ee93721-ee1c-4382-bebf-7c4775eea35b', 0.27), ('ba54b1f0-2464-4e41-999e-43cdfcfb0069', 0.26), ('f02c0773-8347-484f-aa59-8c1d7d4eeb3b', 0.25), ('f3725ee2-5629-4ae4-8e34-d24636246e72', 0.24), ('c376ad0a-c4d7-4e25-9846-12f317adb86b', 0.24), ('584325d2-07f9-42d3-8256-ec979c81d018', 0.24), ('ec6073bd-460a-4fdd-9b5b-de47494c4eb5', 0.23), ('d1155075-a135-49a8-b84f-4e97b6a77cda', 0.23), ('cd77b8a0-e166-4c7e-8579-bcf1da0359c3', 0.23), ('a773feca-d3ba-4869-8a96-89fb8e57f26b', 0.23), ('9d77dc00-def9-460b-9ca4-d01f57d99229', 0.23), ('c86093ac-d52b-4252-92e9-e96f7c2845a2', 0.22), ('461bf650-9229-448c-9e7f-f02431b3900a', 0.22), ('78c60cd9-a948-48a8-adb3-d6d0a20cb703', 0.21), ('3123cb6a-ee7b-47ea-a282-be8a49446aaf', 0.21), ('0b2ac1f5-4d23-4f02-bf28-51365fb99025', 0.21), ('3036b43b-0b3c-4413-a48a-6ec35e213f00', 0.2), ('01cacc3b-4124-4d14-bef8-b8fc96f9078c', 0.2), ('fdb9777c-e04d-4775-b201-9be962ee9482', 0.19), ('d860183d-a7a9-4ebe-b9e2-a414067cbe9c', 0.19), ('c99543cb-489a-48d9-a32e-f4c4fa38d274', 0.19), ('7d1435a0-0f6d-41da-a738-e658d5a47f62', 0.19), ('41e808c3-d362-4b93-acc7-88631d1c73cb', 0.19), ('1f9bbd03-5fe7-416c-a660-8455b4e16e94', 0.19), ('1ec151bd-9e7b-4f88-bf9b-78dda6312402', 0.19), ('02b89e63-35ff-41b7-a8e0-e827dca5b324', 0.19), ('df99e42d-38ee-46f5-9508-319414d72ff7', 0.18), ('d635989b-7c5f-4367-8109-402717008417', 0.18), ('cebec4f1-d885-48d4-831e-32a8da87d9b2', 0.18), ('ce3d75ee-737c-48fe-a164-5f11d5f83446', 0.18), ('a5882a6e-6b23-43d1-9105-ae72e11c261b', 0.18), ('972d4278-e898-4481-bc86-192b1e733ed3', 0.18), ('58bfe4da-fbec-445a-b27c-94c8d9a08732', 0.18), ('5495c515-6021-44ed-b5bd-a754ab2af55c', 0.18), ('17343d98-3d0a-48fb-9bc5-e2897f42d7d5', 0.18), ('16ccf8c5-a948-4fa9-ac0b-b1a2d06caf37', 0.18), ('00203a79-66b2-4dee-bfd2-2e4940f14ea8', 0.18), ('f3e7ee69-61f6-4113-90f0-245844bcef31', 0.17), ('ec7b6c99-f2e0-435e-b9bf-3aedb6281f43', 0.17), ('e057ab50-2e78-4a25-981d-fab08cd94469', 0.17), ('dbdea299-2d8e-423d-8fd3-81502f3ffb9a', 0.17), ('d937eb62-17d6-42cf-8570-7de0bac45c44', 0.17), ('d701b327-18ba-4ce5-8e47-41471818c7a8', 0.17), ('ce2accb5-7b89-4f90-be49-792e52e34059', 0.17), ('cd69f9a3-4532-49db-8816-e2adc25973e2', 0.17), ('c1a51773-f45b-4b43-bd9a-5805eef21738', 0.17), ('b113861d-9b3a-4443-a9fc-391d1f54191a', 0.17), ('ac65dedb-d39d-4aab-8b43-c770ea3efeb4', 0.17), ('a9b04843-9083-4baa-bb2e-554049cd038f', 0.17), ('8d4ba367-1949-4921-86fe-22d7aaa3c074', 0.17), ('733db5df-fba0-4157-9b24-b2aaf7c09a07', 0.17), ('569b5499-b6de-41d7-836e-e0f7f76c6681', 0.17), ('449b8f72-1c23-4da4-9cc8-342592fe83f5', 0.17), ('3b98cf4d-9598-4a69-9aba-278b580e5b90', 0.17), ('264196f0-4e24-4dda-ab1c-31e893f0c5e5', 0.17), ('255c0b34-8683-4521-b140-f6370cc29967', 0.17), ('233c0d83-f746-4c6e-a277-a2dd0fa83444', 0.17), ('216b8ec4-a8a0-40ba-8929-07701a9e0221', 0.17), ('1fa12cbc-c7be-41da-a520-b4656eaafa78', 0.17), ('1ec2cbae-d4e7-4876-b2c9-a18d0ec0bc31', 0.17), ('163ac1fe-aaab-4c45-89ae-f82b3aefe25c', 0.17), ('1042fa87-b911-4d2d-8d43-c0f125970c62', 0.17), ('07c14f2c-11a3-430a-bff0-81af4656b422', 0.17), ('04512eb5-90d6-4f66-b3b2-e68b0ec35c50', 0.17), ('ff732dcb-cd7d-479a-8ae7-830d6ad3b30c', 0.16), ('fc8d1f37-613b-406f-96a5-ed4a575390e5', 0.16), ('f822d0a4-0910-4e65-854c-63dd101662d0', 0.16), ('f3707ebd-e1e1-495e-a524-8fd2b7ed03e9', 0.16), ('e8e1727b-0881-4fd9-810f-d8f3d5c2bf42', 0.16), ('ccff4b25-1bf6-421e-ae1f-477e27699ac2', 0.16), ('c3d37157-2c7c-4a92-a4d2-cebe45fd5512', 0.16), ('b36a74aa-5a26-41f0-9e84-29804f36dafb', 0.16), ('abaef725-43a4-4797-b3aa-f9cef44a754d', 0.16), ('a9e36fdb-6584-41be-82c4-def74d497530', 0.16), ('9884012e-6093-4153-bbdb-cab7faabc979', 0.16), ('966f3dbd-82a8-443a-ad28-3ed608b3ab86', 0.16), ('832737ae-b3a8-4617-82b8-e14ff863c80e', 0.16), ('6384ccb9-d435-4bd5-a029-89f464be7c91', 0.16), ('4fdd84d4-4ed7-4758-9c11-4936878b08b4', 0.16), ('46eb974d-8a4d-413f-9a74-daecb36cb313', 0.16), ('4061aa9b-6fce-47c2-8379-8e6e94c80574', 0.16), ('328b3f68-6357-4f93-9faf-03dbaf38c82e', 0.16), ('2a279d34-d7d3-4d8d-b405-e6baee01691a', 0.16), ('21b0ecb7-5434-4658-8370-df16cb978b8e', 0.16), ('f2bc8181-c2d2-4520-965e-74af3a3ec1ec', 0.15), ('ee8b17ee-54c1-485f-90b7-014efbeb854d', 0.15), ('e2b435d6-6881-4c45-831a-9d305aea51ae', 0.15), ('dc65d97f-8b9e-41d6-807f-97e5ccd3e5cb', 0.15), ('d8db3af9-a16b-4733-8190-bc45c1f36c24', 0.15), ('d61347a0-f1bf-49cc-b9e3-0a9c0a1eb696', 0.15), ('d4cc1912-436c-4e65-b0e9-ac83c87758ba', 0.15), ('d3d1e87c-4a82-4805-8e09-11b8f78d78d0', 0.15), ('d08687b8-ee43-4b29-b188-d3dced719a9e', 0.15), ('ce93e481-3c96-42a1-9041-cdc98dd2133b', 0.15), ('cdc5181c-ef72-4916-be87-61ae420a151f', 0.15), ('caebbc08-57f2-4646-a7e2-6ad0514170a5', 0.15), ('caa351a7-93f8-4c3f-bc2c-7688823e7bb0', 0.15), ('c834a5ff-ac31-453f-8292-acaa173a08b1', 0.15), ('c75fc6b0-2a74-44dc-9e96-b1314e30f303', 0.15), ('c6f5a109-3aa8-4323-bef2-0d07ceafeff9', 0.15), ('b6e40c85-d6e5-4eae-9f5d-e1fce725b06b', 0.15), ('b6430374-7f97-4c8e-9d19-1dece12a29da', 0.15), ('b3d8c319-216f-4b71-9bfa-7009143e2809', 0.15), ('a9f8a0b3-ffa1-41b8-a0b0-e051d03ebba0', 0.15), ('a9f03483-6b8b-4295-a28c-713b738297ab', 0.15), ('a9c32051-851b-47ec-9f2f-f23e49bc4b34', 0.15), ('98133c0b-c9e2-407f-976a-838f69e09276', 0.15), ('924d1b77-7f0d-4fd6-832e-4ff197fdcd92', 0.15), ('92381b04-4abf-4abd-ad79-ddd61fbb4b2d', 0.15), ('8e503f89-c8b2-4172-b1bc-abb7bd49bf3a', 0.15),
               ('8c50f67e-11fd-46dd-a4f0-90c88d08d0c4', 0.15), ('7eea8da5-7caf-424c-9418-64ab4ec298b8', 0.15), ('6f25994d-843d-4ac6-ac1f-ab58eb49e1e5', 0.15), ('52882c2c-735c-457b-8798-310df26304c3', 0.15), ('4f71351e-8b7d-40ce-8f20-be53076628e9', 0.15), ('451a6f6a-f2cb-4f3b-a355-5f036126b76b', 0.15), ('4419a497-e6b0-4d8f-b7a7-922e15e2b5f9', 0.15), ('3a2d199f-48b2-435e-9ebe-b288d6e0bbc5', 0.15), ('35f96b3f-44d3-4ca3-946c-9e70a24164f7', 0.15), ('33e7993a-6f67-4eed-8273-593927d0b1d1', 0.15), ('31842533-a35a-445b-ba43-b248b4b8fecf', 0.15), ('2acb28f4-ca91-48a4-8844-38b160f0984f', 0.15), ('22c914ce-8c01-4e53-8069-d58a7cde2e77', 0.15), ('18f780b8-5dcb-443f-9dcc-fe513a045b95', 0.15), ('18d6b709-12cb-4b78-a136-ea1789db3d55', 0.15), ('0c038fb3-e907-4d2e-bfb9-3a0e492d64ff', 0.15), ('0b298724-70f8-4d33-a6e1-c23cfa979f9a', 0.15), ('04bd6856-5769-48b8-b71a-acf8c2670623', 0.15), ('0193fe7c-25c3-412b-8e1c-d313852cd59c', 0.15), ('fbd3bd7b-5850-4b21-8cf8-4e0ca3d023ae', 0.14), ('fa0919ec-d640-4dec-a43c-b223d2798022', 0.14), ('f8a8b81f-7f1a-4928-8d3e-e8fb8684c538', 0.14), ('eef51be1-8468-4460-9b3c-24da541b2bb5', 0.14), ('e73db76b-074d-4cf7-aa5b-203e5d1b6893', 0.14), ('dc5f39c5-a4ba-4191-96a8-f54dc075b297', 0.14), ('c2f499e9-9a54-4bb3-8e5c-af50aceafa61', 0.14), ('c02a3dfa-3445-4c8c-be25-e14be529bfed', 0.14), ('bbcdc479-874a-45c7-9427-4a0b794f148f', 0.14), ('b73798bd-a9a2-4f92-be99-4ceafcf1f78f', 0.14), ('b38e363a-ba13-4df9-8cc1-f8b1ea591718', 0.14), ('ab110e7b-5446-4666-966a-dbe6a0cbee61', 0.14), ('aaf78cef-282c-4fbb-93ab-6d61d7d36e75', 0.14), ('9c16a841-5805-4f51-832d-b8c5ecad03e0', 0.14), ('98a3f34a-8309-4464-982b-69d85ad56e04', 0.14), ('957e19cc-824c-4204-a5f9-d30f2d34229c', 0.14), ('8a6f4553-3900-4d99-be1a-66c2c90101f2', 0.14), ('84975286-1254-4d21-8009-a90304f05232', 0.14), ('837a04b3-5d99-4815-9e90-2ddbbac20212', 0.14), ('80f9a5d9-d5d9-4d06-8051-171be172ad68', 0.14), ('7e5c4547-153b-42d5-b2e8-abe134d7cb38', 0.14), ('6e142a8c-ad18-4199-be8c-aded7c79cd21', 0.14), ('67cccca3-595e-4eaa-9998-c254e845fe73', 0.14), ('679aaece-4555-4b7f-b3e6-f2f707e81480', 0.14), ('58e54316-15fe-4211-8e8f-7611ae759dab', 0.14), ('519f719f-7834-41f4-bde6-df27827986c0', 0.14), ('4d311ace-c409-493f-b01c-350ed6b328bf', 0.14), ('47f0dec6-1375-47e8-b812-c77a146755cb', 0.14), ('408b3790-3366-48d7-af05-65c2e65a5470', 0.14), ('3e927563-c9ad-46b3-a1e3-0adbc16e9926', 0.14), ('2238d2de-cb84-459d-a4df-d5950fc7c342', 0.14), ('101a4118-b0b9-49ea-a848-5acccf419b74', 0.14), ('f61f58e6-5d1b-43a3-9b24-28036421543b', 0.13), ('cadcd4c3-efaa-438f-93ba-1434ca5e0190', 0.13), ('699bd0da-5909-4b98-ae8c-39c8c13a1ecf', 0.13), ('53877802-17b5-40ec-a801-653ff4def6cb', 0.13), ('50fb5e11-3610-4363-aff2-2967db1112b6', 0.13), ('33b11038-799c-4c23-97b6-16b526284639', 0.13), ('2ed42572-c6c9-43a4-94bb-b7399e0242f7', 0.13), ('2e1e6f7f-55ec-463d-a3af-013a711c9d1c', 0.13), ('fdaaab2f-6f82-4347-b6ab-1b5d482f6a71', 0.12), ('f45a0d4d-3ff9-4232-a0bc-1f411a63e200', 0.12), ('f1df1849-6365-4ebf-a47a-1db232b847d1', 0.12), ('f1c7786a-a42c-49ad-9a40-1a2df9a5178f', 0.12), ('ed7120f3-6692-429e-90cf-91eb445ae9d7', 0.12), ('eaea02e2-59fb-4abe-b8e6-ac87c983fe8f', 0.12), ('e79261d3-8a05-4924-a699-88a19cc7441a', 0.12), ('e4da42df-287e-4138-93ce-69efd9b8b104', 0.12), ('df81772f-41e2-4ec7-ab10-0bebf3c8230e', 0.12), ('dc125892-cce9-44f2-a39b-fc6755e31a4f', 0.12), ('db6c75d5-f52e-4823-9417-0bc8e7ba6ad0', 0.12), ('cff85e89-70f8-484c-9ddf-9baaaf60f68b', 0.12), ('cfee4373-a0ff-4ccd-82e2-573e23e8eae7', 0.12), ('c027bf72-c9b7-4553-8d79-6fe967d88d38', 0.12), ('bfd9407c-8501-41d8-a259-9f252350ea8e', 0.12), ('b7248ef4-3589-496f-b293-b3950db6df32', 0.12), ('aa7a25fc-3c99-40a4-a21e-a66df8d31d57', 0.12), ('a0ffe000-bfcc-4e95-9b5f-cf2247387edb', 0.12), ('a080f46d-3b44-429a-a23d-87a3dc6b7c23', 0.12), ('9dbc74ed-224a-4d9a-b3d4-1d84b88425b9', 0.12), ('95a825ce-089b-4017-9595-93ef812ef65f', 0.12), ('8610f7d4-cea3-4c37-9658-95911198b653', 0.12), ('845834d9-82bd-4995-8337-9810bf157a0f', 0.12), ('81606390-0408-424b-9359-fe33987110f6', 0.12), ('80aa352f-c69a-4253-b3d4-da89a52bd09c', 0.12), ('7824c9df-345a-4891-b49c-198870795fb5', 0.12), ('75fc5749-136c-4a3d-a8b3-476a07376f6d', 0.12), ('69c630ca-1ae5-4818-8cf1-6a2bf3cee5de', 0.12), ('67313c62-7e90-4d05-a46a-6d6fd9ec0c89', 0.12), ('5e7106b5-2d2b-4500-ac7c-4db9315d5510', 0.12), ('5da6c100-f94d-43d6-91ee-6ed4c5f1bfd1', 0.12), ('5a577e90-7a1c-49be-81f4-33c728a7d3e3', 0.12), ('590cf128-834c-4ec1-9a2f-284602a4f4cb', 0.12), ('542dea5d-4d41-4a2a-9aea-a469c191c25b', 0.12), ('53fae02b-60bd-4bfd-990d-c2b8267b3634', 0.12), ('53de9e70-80be-42c8-a50e-170b9444522b', 0.12), ('52518c08-083b-4d0f-9da3-7a5d86be1abb', 0.12), ('4a63eb4e-0ddb-4542-b880-0fea516de971', 0.12), ('46b60e23-1549-40da-a60d-0354915bcf56', 0.12), ('3ad6ec7b-c134-454e-9be9-9c3f4ed9fc27', 0.12), ('325b9452-ff1c-4a54-a46d-cb6db2b4ca1a', 0.12), ('30b62513-084d-473f-8f78-8de942fff232', 0.12), ('2d14f048-319c-43ae-8f41-cb648e245887', 0.12), ('22e195a3-0e82-46c3-86d7-f4e2e587dcf0', 0.12), ('1e4467eb-665a-4057-93c6-aaa193cd4a71', 0.12), ('1b47a6d2-e19c-49ac-a96c-115b5521aeeb', 0.12), ('1ab26022-cfbe-4ae4-8924-ec973b0c1495', 0.12), ('1a912fea-cf77-470c-b839-8a77651f5c5d', 0.12), ('16065dee-f3d4-4e9b-8b1c-aa219ea632fe', 0.12), ('1037c6c0-7c15-4725-9ae3-e2919907b16f', 0.12), ('0da394cb-1714-4a0a-b10b-dd364f2c66e1', 0.12), ('fa1487a4-9294-421c-879a-3388149be7a9', 0.11), ('f8624528-f6b7-4b4a-9044-8583443177b6', 0.11), ('e2049be1-5b0f-4f15-8a3f-08620969b291', 0.11), ('e162ae90-29bc-49eb-ab54-b251eb233a00', 0.11), ('e110098f-a3f0-4487-a25d-100b374d87e7', 0.11), ('ded8721a-96e9-4a17-891b-2cab61cf87a8', 0.11), ('d4f1e1a2-f986-421a-bc0e-2ca4f344c14c', 0.11), ('d35e0f18-1a20-4ab9-b27c-373326fa4a86', 0.11), ('be6fcf9f-060a-41d1-ac34-be97b9d72f10', 0.11), ('be3f4e97-f8fd-45af-ad7e-9911fa79e877', 0.11), ('b99014d5-ca72-4c02-9755-6d0750314967', 0.11), ('adb1d718-9fcb-4ec2-9ff6-db1d4036905d', 0.11), ('a6a8e61d-61ee-4f9b-a594-5a29a67bfe39', 0.11), ('9d0fb0f8-5e1f-4d4b-a7b3-6558f50d3ee7', 0.11), ('9cac122b-1b79-44ac-9b03-c3d3a94de94f', 0.11), ('86be94a6-9836-4a12-b1d8-3faf638fae55', 0.11), ('7f0b5b75-cb43-4c0e-b572-e6090ad5e235', 0.11), ('7be94649-6b24-428f-8197-8c0cf52cc961', 0.11), ('6a94a266-7409-4a2d-90cb-237e84b5dc5e', 0.11), ('64f5fa88-e355-4ad3-871f-938e1d8202a9', 0.11), ('61149edf-28bc-4d44-81a5-daaadfc31738', 0.11), ('582b36b5-5f6f-4329-8958-d1cc1225adf2', 0.11), ('4c9d5c46-b8d1-4b3e-9cf5-86cc36eaa8ab', 0.11), ('489376f4-8864-470c-a111-7bfc7e4bd51a', 0.11), ('3631f22a-8a04-427a-9a30-5ee16790dc21', 0.11), ('2fd5c378-9abc-4d4a-9674-9c95c1c226e2', 0.11), ('2d75b05d-a88d-4fa2-ab28-9a38df1589dc', 0.11), ('2550ac17-0612-4f61-a73f-a2e3fbbb2e56', 0.11), ('bbf0b845-b8dc-4eac-9a1d-a05dbbc64200', 0.1), ('a91b69c8-5a83-4a85-9b9b-32ecd4325522', 0.1), ('36324dd7-b639-473e-86bc-7e21630fbb46', 0.1), ('d3689ad8-f975-4e57-9fca-394dce51a799', 0.09), ('c6d70a0f-abf0-4d92-9b08-5df503e92a15', 0.09), ('c1e0b156-1c48-4057-ac1b-856df68ee1fb', 0.09), ('b2ef17dc-9fb2-43ef-82ed-3b1aa9776530', 0.09), ('5a2c6767-c866-480c-be5d-09ba2f8b4567', 0.09), ('2e8feff2-998a-4b77-aff8-40b60421609d', 0.09), ('17a255b9-e807-4caa-8f41-02f443ac579b', 0.09), ('128bd6a1-62a7-4714-9467-e3673fd3f96f', 0.09), ('04dd4d01-d42e-4488-847d-6b92625d9140', 0.09), ('ff567515-f37b-4f84-a548-6fbeb3e1a15f', 0.08), ('f59da4c8-6bb8-49df-aa30-14c0c9a11341', 0.08), ('ebd3b680-d87d-4f58-b05d-05d84509561f', 0.08), ('dbce4896-0327-4e55-bc8d-cca448492e05', 0.08), ('d3461c7f-80be-4d68-9491-70bc0fe418e1', 0.08), 
               ('d18bd367-bc71-41cc-abf8-78671c6144bd', 0.08), ('ca0e126f-e8f1-4472-adbf-ecd5edb9e2b0', 0.08), ('b4c6d34a-88e3-4286-a761-9d5593110906', 0.08), ('b2ba4ab6-f00d-4e1c-bdf6-b1bc1133086c', 0.08), ('b082d3d5-5e7a-4e4c-8e58-b3589979cbfa', 0.08), ('aed6748c-060d-42ec-9184-cb174cdd776a', 0.08), ('a074dd23-7e83-4a62-b86d-b6520463cf59', 0.08), ('9ab0701b-c44a-4b3e-bb95-6b39cdf4b45c', 0.08), ('93480360-59c2-4563-b2d1-fca1d24d3a36', 0.08), ('8eabed22-9a42-4bbc-a88b-b1a62d25bdd1', 0.08), ('869935cb-f634-44ca-b300-f0419fc62ae4', 0.08), ('6bcdd5bc-0e00-47c8-a075-2ef227c58f9a', 0.08), ('68e9a8d5-4de4-4442-8895-4de2fa7fd385', 0.08), ('681c6d52-d29a-4c79-aa4b-7eae5bc98d73', 0.08), ('5614a0ee-d13f-4116-b457-f5dc40371d65', 0.08), ('327d0b29-b299-4f16-9133-61194e06ae0d', 0.08), ('2d638ffd-bc45-496e-8bd5-182b0fe516d5', 0.08), ('090e41ae-e690-46ba-b528-00a2187504dc', 0.08), ('f59fc7ab-5d2f-4ef3-aa63-dabef6a1dceb', 0.07), ('e2c7d8fd-fa59-4223-b2a5-2bf31c01ceb1', 0.07), ('d8acb81d-7668-416c-b323-69f92086dd4f', 0.07), ('cf195407-173e-4741-a23f-ac746062b3ea', 0.07), ('cc0b0695-c488-4561-9874-ba56003d4e61', 0.07), ('cafce511-519e-4b3f-85c1-57ba8c913a88', 0.07), ('b8efbc5a-9fa4-4f11-90db-14b0600dcede', 0.07), ('a0f40b39-66be-4043-8745-49c37c8942f4', 0.07), ('99e5af20-3de6-4b2e-a06b-8d30d5841ede', 0.07), ('9098fdf9-6760-4fb8-ad2e-35f37f6c7c19', 0.07), ('6142943e-711e-426a-916d-aef3744bbb02', 0.07), ('56a4094a-edbb-410c-86f5-017c5b754e12', 0.07), ('276c084e-767d-48f1-b066-cc2ce6682e3a', 0.07), ('1eff7848-1723-406a-b8a3-e31bfb9fb8a8', 0.07), ('15044772-0ad9-4156-b753-cda70d7c205c', 0.07), ('d9ad0806-659a-42d4-a4e7-5626f5191884', 0.06), ('d5a4670f-1b5f-4f0d-93cd-2d17e724f750', 0.06), ('cf347e17-8069-4af9-9cdb-8a0f2b75afb3', 0.06), ('cf0b81b7-5362-481e-957d-3e3350d0dd3b', 0.06), ('c51cf43f-5d26-400f-8ec3-48298770baef', 0.06), ('9d453d95-984f-45e3-bdfe-4f2a28bc1811', 0.06), ('90b36067-6410-4744-b63a-1a53c10b17ad', 0.06), ('83595dd0-62b3-48f7-a71c-7195076f60ad', 0.06), ('5bbf6177-e1cf-4d45-b6d2-215b7d490212', 0.06), ('4b682522-22f6-420f-b44b-34027841e513', 0.06), ('d4469927-0800-4124-a8c2-2ab56fd9a192', 0.04), ('c6807c9b-7c3f-4194-85b1-72b7a63f9f0c', 0.03), ('ac56989f-44f4-4524-8cbd-0d7f798b6b65', 0.03), ('92b24dc4-f41a-4b84-81b4-bc094c6ac345', 0.03), ('596193d7-97b8-4fa8-b7a7-01ec15256c0e', 0.03), ('43b833b9-845b-439f-84d9-a8a45ce4a862', 0.03), ('28f3ec5b-a5e8-493f-a519-8777f50bf871', 0.03), ('ff35442b-f157-4d18-aae7-b28f40a90c1b', 0.02), ('fdb09a43-6443-4066-b331-e8a3d615d147', 0.02), ('f4cd034a-ace0-434f-9685-5f171da3c259', 0.02), ('f4206738-2d9f-407f-b31a-cefd2c4803dc', 0.02), ('eb08087d-c641-40ac-bc53-a725b78cdeb3', 0.02), ('e6749e19-60dd-46f2-bbae-336e3d49b211', 0.02), ('d788ffbd-537f-4626-a217-b77be39cd755', 0.02), ('d24426ef-5e53-489f-96be-54940527616a', 0.02), ('cc7aa018-cca2-41ef-9b90-85a879d464f8', 0.02), ('c34d72fb-aa78-41c2-93d6-fcc162661c29', 0.02), ('b5ef7b88-c8b5-4a11-b1f3-0f3d3415189a', 0.02), ('aadfeb85-cd41-4c3e-8a46-b06cf95d8aad', 0.02), ('9bdc75c5-784f-46b9-9e0a-9f4297b142a5', 0.02), ('95873cda-ef0a-4960-b764-001e9ae2c72b', 0.02), ('936d4bfa-3d1e-4e10-9afd-5c49726c9f50', 0.02), ('86dc68dc-2790-4962-bec1-214cd346fac2', 0.02), ('7a326ba0-780a-4fbd-a720-9e40a3c9761c', 0.02), ('7967705f-8cc8-4f8f-84f2-03e51c00916b', 0.02), ('732a25c4-bd3d-49f9-b207-5e8bbc3f4c4c', 0.02), ('71dbc9d8-9fe7-4d1b-a8c0-adc8f1f3b8e8', 0.02), ('6f3aafc4-ccaa-4328-af43-b58ae86951d4', 0.02), ('60356122-9b1d-4d4d-908c-17038977b4ca', 0.02), ('5633b4c4-e471-4af8-821d-f0583a831abb', 0.02), ('4cfb3e2d-7af2-41b2-adcf-4077b2016b4a', 0.02), ('466b431f-ba76-4294-982a-301b699d7776', 0.02), ('460f5e7d-a7ac-49b7-8acb-ae8076c222fa', 0.02), ('3f220c22-4aa7-425d-87bb-3446bca495a3', 0.02), ('334ef628-f49d-4b69-89bd-f3423c9c39f3', 0.02), ('2cef945b-53f2-40b2-a6ac-0f7d3a4737fb', 0.02), ('232339e9-0065-4d01-b40a-ce029865615d', 0.02), ('1afce6b3-92c3-4cae-939e-a3eec476f3ea', 0.02), ('18d11f2d-ea85-4225-95a5-c601ad932432', 0.02), ('141ab846-d24f-4dc6-bac0-1a0c7bab5395', 0.02), ('13e49cc5-65b6-4da2-9bce-60a12353d18e', 0.02), ('0fa62568-a084-4377-9e1a-c865d7c33ae0', 0.02), ('01c03e2a-0209-4d9a-9e76-79c803e3b3b0', 0.02), ('e5165d7f-36ce-44b5-b3b3-dc4045739c31', 0.01), ('e26ceb58-134a-4b8b-a190-5ffac855521d', 0.01), ('dea63f8b-79d9-4b94-988a-4b0594840bd3', 0.01), ('db6edc21-a42e-4049-a4eb-b8b2fb54984e', 0.01), ('d1e479c0-e9ff-496e-ad78-70a94a9a125b', 0.01), ('d0a2ce2b-4eee-461e-a4b3-8f8e7ce86ed1', 0.01), ('cc25d9ec-68d5-4a95-9c32-880888853a58', 0.01), ('cadf5d4a-7f4b-467e-a425-c2f8f8c53ea9', 0.01), ('c7923ce8-9efd-4ba4-a621-7a37f25c3e6d', 0.01), ('b92ded1b-bbd9-4aa4-bf26-2c66612e0357', 0.01), ('a00e8c56-1b5b-4369-a323-e0cedfeaa1a8', 0.01), ('8b7d5cc3-d056-4b5c-addf-3ad3fe8dc2b4', 0.01), ('850831b9-5093-4256-965d-fae98867334a', 0.01), ('84d6d5e7-1478-4827-8a70-fe3ba4a5a347', 0.01), ('7f32cbf8-cd29-4ee6-84d6-be4edc51f30f', 0.01), ('790d22b5-d1ed-4bc9-8205-22759480b7a3', 0.01), ('73b94708-0fb0-4e7f-bdc8-f53dcb2417ce', 0.01), ('68c29f16-ef9f-4119-870e-da6d99e4004f', 0.01), ('5d73867e-d491-41e8-9561-57c3b7916bc5', 0.01), ('5718c30c-04fb-4c75-a1e0-d3f9a58239c4', 0.01), ('4c26981a-d33c-4978-808d-054269eeb3a9', 0.01), ('43b97bd3-c7cb-4a8f-b91e-6f93d7eb6792', 0.01), ('42efe95e-8d1a-47ad-b414-3b076ba0b211', 0.01), ('3b1ff25f-06ce-437f-ba25-420c539d15c9', 0.01), ('3876520b-a583-48b1-b833-aa66d2c5770c', 0.01), ('376b0ae5-28f1-4ca3-b950-3018243785f1', 0.01), ('311d4367-4acc-491b-98ee-08b94b66560c', 0.01), ('2a3d16d1-c00c-4008-b772-99aba00473c5', 0.01), ('17c6b041-5962-4a5f-9b4d-437cff076264', 0.01), ('154b05cc-9b8d-48e6-afe2-8336c042a135', 0.01), ('1182386f-d653-49f8-946f-b5d0400893f2', 0.01)]

# Load the data from JSON file
file_path = '/Users/paullemaire/Documents/data2024.json'
df = pd.read_json(file_path, lines=True)

# Filter data based on timestamp
df = df[df['Downlink-GTW-Timestamp'] <= '2024-01-29 00:00:00']

# Convert the 'Downlink-GTW-Timestamp' column to datetime
df['Downlink-GTW-Timestamp'] = pd.to_datetime(df['Downlink-GTW-Timestamp'])

# Adding the difference variable
df1 = df.sort_values(['Downlink-CRM-DeviceID', 'Downlink-GTW-Timestamp'])
df1['diff_seconds'] = df1.groupby('Downlink-CRM-DeviceID')['Downlink-GTW-Timestamp'].diff().dt.total_seconds()

# Define the function to pad or truncate sequences
def pad_or_truncate(seq, target_length):
    return seq[:target_length] + [0] * (target_length - len(seq))

# Initialize the autoencoder model
model = Autoencoder()

# Load the saved model state dictionary
model.load_state_dict(torch.load('best_autoencoder2.pth'))
model.eval()

# Define the loss criterion
criterion = nn.MSELoss()

# Compute the loss for each device and store the results
device_losses = {}
for (device_id,CV) in List_of_CVs:
    # Filter the data for the specified device
    df_device = df[df['Downlink-CRM-DeviceID'] == device_id]

    # Create the list of time differences
    time_diff = df_device['Downlink-GTW-Timestamp'].diff().dt.total_seconds().tolist()

    # Remove the initial NaN value
    time_diff = [x for x in time_diff if pd.notna(x)]

    if time_diff:
        # Normalize time_diff
        time_diff = [(i - min(time_diff)) / (max(time_diff) - min(time_diff)) for i in time_diff]

        # Pad or truncate to length 100
        time_diff_padded = pad_or_truncate(time_diff, 100)
        tensor = torch.tensor(time_diff_padded).float().unsqueeze(0)

        # Compute the reconstruction and loss
        with torch.no_grad():
            recon = model(tensor)
            loss = criterion(recon, tensor)
            device_losses[device_id] = loss.item()

# adding the difference variable
df['Downlink-GTW-Timestamp'] = pd.to_datetime(df['Downlink-GTW-Timestamp'])
df = df.sort_values(['Downlink-CRM-DeviceID', 'Downlink-GTW-Timestamp'])
df['diff_seconds'] = df.groupby('Downlink-CRM-DeviceID')['Downlink-GTW-Timestamp'].diff().dt.total_seconds()

final_data = {}
aux_data = []
factor = 2
for i in df['Downlink-CRM-DeviceID'].unique():
    device_data = df[df['Downlink-CRM-DeviceID'] == i]['diff_seconds']
    value = round(device_data.std()/device_data.mean(),2)
    if not value in final_data.keys():
        final_data[value] = 0
    final_data[value] += 1
    aux_data.append((i, value))

final_data = dict(sorted(final_data.items(), key=lambda x: x[0]))
aux_data = list(sorted(aux_data, key=lambda x: x[1]))[::-1]

# Print the total number of devices evaluated
print(f'Total number of devices evaluated: {len(device_losses)}')

# Plot the sorted losses with colors based on CV
plt.figure(figsize=(12, 6))
colors = ['red' if cv > 0.35 else 'blue' for device_id, cv in List_of_CVs]
plt.bar(device_losses.keys(), device_losses.values(), color=colors)
plt.xticks(rotation=90)
plt.xlabel('Device ID')
plt.ylabel('Loss')
plt.title('Reconstruction Loss for Each Device (Sorted)')
plt.show()
